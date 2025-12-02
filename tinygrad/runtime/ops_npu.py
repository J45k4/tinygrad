import os, ctypes, ctypes.util, mmap, functools
from tinygrad.device import Compiled, LRUAllocator, BufferSpec, Compiler
from tinygrad.helpers import getenv
from tinygrad.renderer.cstyle import HIPRenderer

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS, _IOC_DIRBITS = 8, 8, 14, 2
_IOC_NRMASK, _IOC_TYPEMASK = (1 << _IOC_NRBITS) - 1, (1 << _IOC_TYPEBITS) - 1
_IOC_SIZEMASK, _IOC_DIRMASK = (1 << _IOC_SIZEBITS) - 1, (1 << _IOC_DIRBITS) - 1
_IOC_NRSHIFT, _IOC_TYPESHIFT = 0, _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT, _IOC_DIRSHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS, _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2

def _IOC(dirv, typev, nr, size): return (dirv << _IOC_DIRSHIFT) | (ord(typev) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)
def DRM_IOWR(nr, struct): return _IOC(_IOC_READ | _IOC_WRITE, 'd', nr, ctypes.sizeof(struct))
def DRM_IOW(nr, struct): return _IOC(_IOC_WRITE, 'd', nr, ctypes.sizeof(struct))

DRM_COMMAND_BASE = 0x40
DRM_ROCKET_CREATE_BO, DRM_ROCKET_SUBMIT, DRM_ROCKET_PREP_BO, DRM_ROCKET_FINI_BO = 0x00, 0x01, 0x02, 0x03

class drm_gem_close(ctypes.Structure): _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]
DRM_IOCTL_GEM_CLOSE = DRM_IOW(0x09, drm_gem_close)

class drm_rocket_create_bo(ctypes.Structure):
  _fields_ = [("size", ctypes.c_uint32), ("handle", ctypes.c_uint32), ("dma_address", ctypes.c_uint64), ("offset", ctypes.c_uint64)]
DRM_IOCTL_ROCKET_CREATE_BO = DRM_IOWR(DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, drm_rocket_create_bo)

class drm_rocket_prep_bo(ctypes.Structure):
  _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("timeout_ns", ctypes.c_int64)]
DRM_IOCTL_ROCKET_PREP_BO = DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, drm_rocket_prep_bo)

class drm_rocket_fini_bo(ctypes.Structure):
  _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]
DRM_IOCTL_ROCKET_FINI_BO = DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, drm_rocket_fini_bo)

class drm_rocket_task(ctypes.Structure): _fields_ = [("regcmd", ctypes.c_uint32), ("regcmd_count", ctypes.c_uint32)]

class drm_rocket_job(ctypes.Structure):
  _fields_ = [("tasks", ctypes.c_uint64), ("in_bo_handles", ctypes.c_uint64), ("out_bo_handles", ctypes.c_uint64),
              ("task_count", ctypes.c_uint32), ("task_struct_size", ctypes.c_uint32), ("in_bo_handle_count", ctypes.c_uint32),
              ("out_bo_handle_count", ctypes.c_uint32)]

class drm_rocket_submit(ctypes.Structure):
  _fields_ = [("jobs", ctypes.c_uint64), ("job_count", ctypes.c_uint32), ("job_struct_size", ctypes.c_uint32), ("reserved", ctypes.c_uint64)]
DRM_IOCTL_ROCKET_SUBMIT = DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, drm_rocket_submit)

class RocketBO:
  def __init__(self, driver:'RocketAccelDriver', handle:int, dma_addr:int, size:int, mm:mmap.mmap|None):
    self.driver, self.handle, self.dma_addr, self.size, self.mm = driver, handle, dma_addr, size, mm

class RocketAccelDriver:
  def __init__(self, device_path:str|None=None):
    self.dev_path = device_path or getenv("NPU_DRM_DEV", "/dev/dri/renderD128")
    self.fd = os.open(self.dev_path, os.O_RDWR | os.O_CLOEXEC)

  def _ioctl(self, req:int, obj:ctypes.Structure):
    ret = libc.ioctl(self.fd, ctypes.c_ulong(req), ctypes.byref(obj))
    if ret != 0:
      errno = ctypes.get_errno()
      raise OSError(errno, os.strerror(errno))
    return obj

  def create_bo(self, size:int) -> RocketBO:
    st = drm_rocket_create_bo(size=size)
    self._ioctl(DRM_IOCTL_ROCKET_CREATE_BO, st)
    mm = mmap.mmap(self.fd, size, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_SHARED, offset=st.offset)
    return RocketBO(self, st.handle, st.dma_address, size, mm)

  def prep_bo(self, handle:int, timeout_ns:int=1_000_000_000): self._ioctl(DRM_IOCTL_ROCKET_PREP_BO, drm_rocket_prep_bo(handle, 0, timeout_ns))
  def fini_bo(self, handle:int): self._ioctl(DRM_IOCTL_ROCKET_FINI_BO, drm_rocket_fini_bo(handle, 0))

  def destroy_bo(self, bo:RocketBO):
    if bo.mm is not None: bo.mm.close()
    self._ioctl(DRM_IOCTL_GEM_CLOSE, drm_gem_close(bo.handle, 0))

  def copy_htod(self, bo:RocketBO, src:memoryview):
    self.prep_bo(bo.handle)
    bo.mm.seek(0)
    bo.mm.write(src)
    self.fini_bo(bo.handle)

  def copy_dtoh(self, dest:memoryview, bo:RocketBO):
    self.prep_bo(bo.handle)
    bo.mm.seek(0)
    dest[:] = bo.mm.read(len(dest))
    self.fini_bo(bo.handle)

  def submit_regcmd(self, regcmd_bo:RocketBO, regcmd_count:int, in_handles:list[int]=None, out_handles:list[int]=None):
    task = drm_rocket_task(regcmd=regcmd_bo.dma_addr, regcmd_count=regcmd_count)
    tasks = (drm_rocket_task * 1)(task)
    in_arr = (ctypes.c_uint32 * len(in_handles))(*in_handles) if in_handles else None
    out_arr = (ctypes.c_uint32 * len(out_handles))(*out_handles) if out_handles else None
    job = drm_rocket_job(tasks=ctypes.addressof(tasks),
                         in_bo_handles=ctypes.addressof(in_arr) if in_arr else 0,
                         out_bo_handles=ctypes.addressof(out_arr) if out_arr else 0,
                         task_count=1, task_struct_size=ctypes.sizeof(drm_rocket_task),
                         in_bo_handle_count=len(in_handles) if in_handles else 0,
                         out_bo_handle_count=len(out_handles) if out_handles else 0)
    jobs = (drm_rocket_job * 1)(job)
    self._ioctl(DRM_IOCTL_ROCKET_SUBMIT, drm_rocket_submit(jobs=ctypes.addressof(jobs), job_count=1,
                                                           job_struct_size=ctypes.sizeof(drm_rocket_job), reserved=0))

  def synchronize(self): pass  # submit is blocking; no-op hook for API parity
  def __del__(self): 
    if hasattr(self, "fd"): os.close(self.fd)

class NPUCompiler(Compiler):
  def __init__(self): super().__init__("compile_npu")
  def compile(self, src:str) -> bytes: return src.encode()

class NPUProgram:
  def __init__(self, dev:'NPUDevice', name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.regcmd_bo = self.dev.driver.create_bo(len(lib))
    self.dev.driver.copy_htod(self.regcmd_bo, memoryview(lib))
    self.regcmd_count = (len(lib) + 3) // 4

  def __del__(self):
    if hasattr(self, "regcmd_bo"): self.dev.driver.destroy_bo(self.regcmd_bo)

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int, ...]=(), wait=False):
    # Inputs/outputs are not wired yet; launch just submits the regcmd buffer.
    self.dev.driver.submit_regcmd(self.regcmd_bo, self.regcmd_count)
    if wait: self.dev.synchronize(); return 0.0

class NPUAllocator(LRUAllocator['NPUDevice']):
  def _alloc(self, size:int, options:BufferSpec): return self.dev.driver.create_bo(size)
  def _free(self, opaque:RocketBO, options:BufferSpec): self.dev.driver.destroy_bo(opaque)
  def _copyin(self, dest:RocketBO, src:memoryview): self.dev.driver.copy_htod(dest, src)
  def _copyout(self, dest:memoryview, src:RocketBO): self.dev.driver.copy_dtoh(dest, src)

class NPUDevice(Compiled):
  def __init__(self, device:str=""):
    dev_path = getenv("NPU_DRM_DEV") or None
    self.driver = RocketAccelDriver(dev_path)
    compilers = [(functools.partial(HIPRenderer, "npu"), NPUCompiler)]
    super().__init__(device if device else "NPU", NPUAllocator(self), compilers, functools.partial(NPUProgram, self))
  def synchronize(self): self.driver.synchronize()
