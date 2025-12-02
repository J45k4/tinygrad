from __future__ import annotations
import os, ctypes, ctypes.util, mmap, functools, json, math
from dataclasses import dataclass, field
from typing import List
from tinygrad.device import Compiled, LRUAllocator, BufferSpec, Compiler
from tinygrad.helpers import getenv
from tinygrad.renderer.cstyle import HIPRenderer

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# **************** IOCTL helpers ****************

_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS, _IOC_DIRBITS = 8, 8, 14, 2
_IOC_NRSHIFT, _IOC_TYPESHIFT = 0, _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT, _IOC_DIRSHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS, _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2

def _IOC(dirv, typev, nr, size):
  return (dirv << _IOC_DIRSHIFT) | (ord(typev) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)
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

# **************** Register helpers (minimal set) ****************

def _ms(val, shift, mask): return (val << shift) & mask

# Constants from rkt_registers ranges
PC_BASE = 0x0000
CNA_BASE = 0x1000
CORE_BASE = 0x3000
DPU_BASE = 0x4000
DPU_RDMA_BASE = 0x5000
PPU_BASE = 0x6000
PPU_RDMA_BASE = 0x7000

def rkt_get_target(reg:int) -> int:
  # Block select derived from base address window (best effort).
  if reg >= PPU_RDMA_BASE: return 7
  if reg >= PPU_BASE: return 6
  if reg >= DPU_RDMA_BASE: return 5
  if reg >= DPU_BASE: return 4
  if reg >= CORE_BASE: return 3
  if reg >= CNA_BASE: return 1
  return 0

REG_PC_OPERATION_ENABLE = 0x00000008
REG_PC_BASE_ADDRESS = 0x00000010
REG_PC_REGISTER_AMOUNTS = 0x00000014

REG_CNA_CBUF_CON0 = 0x00001040
REG_CNA_CBUF_CON1 = 0x00001044
REG_CNA_CONV_CON1 = 0x0000100c
REG_CNA_CONV_CON2 = 0x00001010
REG_CNA_CONV_CON3 = 0x00001014
REG_CNA_DATA_SIZE0 = 0x00001020
REG_CNA_DATA_SIZE1 = 0x00001024
REG_CNA_DATA_SIZE2 = 0x00001028
REG_CNA_DATA_SIZE3 = 0x0000102c
REG_CNA_WEIGHT_SIZE0 = 0x00001030
REG_CNA_WEIGHT_SIZE1 = 0x00001034
REG_CNA_WEIGHT_SIZE2 = 0x00001038
REG_CNA_CVT_CON0 = 0x0000104c
REG_CNA_CVT_CON1 = 0x00001050
REG_CNA_CVT_CON2 = 0x00001054
REG_CNA_CVT_CON3 = 0x00001058
REG_CNA_CVT_CON4 = 0x0000105c
REG_CNA_CVT_CON5 = 0x00001180
REG_CNA_PAD_CON0 = 0x00001068
REG_CNA_PAD_CON1 = 0x00001184
REG_CNA_FEATURE_DATA_ADDR = 0x00001070
REG_CNA_DMA_CON0 = 0x00001078
REG_CNA_DMA_CON1 = 0x0000107c
REG_CNA_DMA_CON2 = 0x00001080
REG_CNA_FC_DATA_SIZE0 = 0x00001084
REG_CNA_FC_DATA_SIZE1 = 0x00001088
REG_CNA_DCOMP_REGNUM = 0x00001104
REG_CNA_DCOMP_CTRL = 0x00001100
REG_CNA_DCOMP_ADDR0 = 0x00001110
REG_CNA_DCOMP_AMOUNT0 = 0x00001140
REG_CNA_DCOMP_AMOUNT1 = 0x00001144
REG_CNA_DCOMP_AMOUNT2 = 0x00001148
REG_CNA_DCOMP_AMOUNT3 = 0x0000114c
REG_CNA_DCOMP_AMOUNT4 = 0x00001150
REG_CNA_DCOMP_AMOUNT5 = 0x00001154
REG_CNA_DCOMP_AMOUNT6 = 0x00001158
REG_CNA_DCOMP_AMOUNT7 = 0x0000115c
REG_CNA_DCOMP_AMOUNT8 = 0x00001160
REG_CNA_DCOMP_AMOUNT9 = 0x00001164
REG_CNA_DCOMP_AMOUNT10 = 0x00001168
REG_CNA_DCOMP_AMOUNT11 = 0x0000116c
REG_CNA_DCOMP_AMOUNT12 = 0x00001170
REG_CNA_DCOMP_AMOUNT13 = 0x00001174
REG_CNA_DCOMP_AMOUNT14 = 0x00001178
REG_CNA_DCOMP_AMOUNT15 = 0x0000117c

REG_CORE_MISC_CFG = 0x00003010
REG_CORE_DATAOUT_SIZE_0 = 0x00003014
REG_CORE_DATAOUT_SIZE_1 = 0x00003018
REG_CORE_CLIP_TRUNCATE = 0x0000301c

REG_DPU_FEATURE_MODE_CFG = 0x0000400c
REG_DPU_DATA_FORMAT = 0x00004010
REG_DPU_OFFSET_PEND = 0x00004014
REG_DPU_DST_BASE_ADDR = 0x00004020
REG_DPU_DST_SURF_STRIDE = 0x00004024
REG_DPU_DATA_CUBE_WIDTH = 0x00004030
REG_DPU_DATA_CUBE_HEIGHT = 0x00004034
REG_DPU_DATA_CUBE_NOTCH_ADDR = 0x00004038
REG_DPU_DATA_CUBE_CHANNEL = 0x0000403c
REG_DPU_BS_CFG = 0x00004040
REG_DPU_BS_ALU_CFG = 0x00004044
REG_DPU_BS_MUL_CFG = 0x00004048
REG_DPU_BS_RELUX_CMP_VALUE = 0x0000404c
REG_DPU_BS_OW_CFG = 0x00004050
REG_DPU_BS_OW_OP = 0x00004054
REG_DPU_WDMA_SIZE_0 = 0x00004058
REG_DPU_WDMA_SIZE_1 = 0x0000405c
REG_DPU_BN_CFG = 0x00004060
REG_DPU_BN_ALU_CFG = 0x00004064
REG_DPU_BN_MUL_CFG = 0x00004068
REG_DPU_BN_RELUX_CMP_VALUE = 0x0000406c
REG_DPU_EW_CFG = 0x00004070
REG_DPU_EW_CVT_OFFSET_VALUE = 0x00004074
REG_DPU_EW_CVT_SCALE_VALUE = 0x00004078
REG_DPU_EW_RELUX_CMP_VALUE = 0x0000407c
REG_DPU_OUT_CVT_OFFSET = 0x00004080
REG_DPU_OUT_CVT_SCALE = 0x00004084
REG_DPU_OUT_CVT_SHIFT = 0x00004088
REG_DPU_EW_OP_VALUE_0 = 0x00004090
REG_DPU_EW_OP_VALUE_1 = 0x00004094
REG_DPU_EW_OP_VALUE_2 = 0x00004098
REG_DPU_EW_OP_VALUE_3 = 0x0000409c
REG_DPU_EW_OP_VALUE_4 = 0x000040a0
REG_DPU_EW_OP_VALUE_5 = 0x000040a4
REG_DPU_EW_OP_VALUE_6 = 0x000040a8
REG_DPU_EW_OP_VALUE_7 = 0x000040ac
REG_DPU_SURFACE_ADD = 0x000040c0

REG_DPU_RDMA_RDMA_S_POINTER = 0x00005004
REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH = 0x0000500c
REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT = 0x00005010
REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL = 0x00005014
REG_DPU_RDMA_RDMA_SRC_BASE_ADDR = 0x00005018
REG_DPU_RDMA_RDMA_BRDMA_CFG = 0x0000501c
REG_DPU_RDMA_RDMA_BS_BASE_ADDR = 0x00005020
REG_DPU_RDMA_RDMA_NRDMA_CFG = 0x00005028
REG_DPU_RDMA_RDMA_BN_BASE_ADDR = 0x0000502c
REG_DPU_RDMA_RDMA_ERDMA_CFG = 0x00005034
REG_DPU_RDMA_RDMA_EW_BASE_ADDR = 0x00005038
REG_DPU_RDMA_RDMA_EW_SURF_STRIDE = 0x00005040
REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG = 0x00005044
REG_DPU_RDMA_RDMA_SRC_DMA_CFG = 0x00005048
REG_DPU_RDMA_RDMA_SURF_NOTCH = 0x0000504c
REG_DPU_RDMA_RDMA_PAD_CFG = 0x00005064
REG_DPU_RDMA_RDMA_WEIGHT = 0x00005068
REG_DPU_RDMA_RDMA_EW_SURF_NOTCH = 0x0000506c

DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT = 3
DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT = 2
DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT = 1

def DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(x): return _ms(x, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, 0x8)
def DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(x): return _ms(x, DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, 0x4)
def DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(x): return _ms(x, DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, 0x2)

def CNA_CBUF_CON0_WEIGHT_BANK(x): return _ms(x, 4, 0x000000f0)
def CNA_CBUF_CON0_DATA_BANK(x): return _ms(x, 0, 0x0000000f)
def CNA_CBUF_CON0_WEIGHT_REUSE(x): return _ms(x, 13, 0x00002000)
def CNA_CONV_CON1_NONALIGN_DMA(x): return _ms(x, 30, 0x40000000)
def CNA_CONV_CON1_GROUP_LINE_OFF(x): return _ms(x, 29, 0x20000000)
def CNA_CONV_CON1_ARGB_IN(x): return _ms(x, 12, 0x0000f000)
def CNA_CONV_CON1_CONV_MODE(x): return _ms(x, 0, 0x0000000f)
def CNA_CONV_CON2_FEATURE_GRAINS(x): return _ms(x, 4, 0x00003ff0)
def CNA_CONV_CON3_CONV_X_STRIDE(x): return _ms(x, 0, 0x00000007)
def CNA_CONV_CON3_CONV_Y_STRIDE(x): return _ms(x, 3, 0x00000038)
def CNA_DATA_SIZE0_DATAIN_WIDTH(x): return _ms(x, 16, 0x07ff0000)
def CNA_DATA_SIZE0_DATAIN_HEIGHT(x): return _ms(x, 0, 0x000007ff)
def CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(x): return _ms(x, 16, 0x3fff0000)
def CNA_DATA_SIZE1_DATAIN_CHANNEL(x): return _ms(x, 0, 0x0000ffff)
def CNA_DATA_SIZE2_DATAOUT_WIDTH(x): return _ms(x, 0, 0x000007ff)
def CNA_DATA_SIZE3_DATAOUT_ATOMICS(x): return _ms(x, 0, 0x003fffff)
def CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(x): return _ms(x, 24, 0x1f000000)
def CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(x): return _ms(x, 16, 0x001f0000)
def CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(x): return _ms(x, 0, 0x00003fff)
def CNA_CBUF_CON1_DATA_ENTRIES(x): return _ms(x, 0, 0x00003fff)
def CNA_CVT_CON0_CVT_TRUNCATE_3(x): return _ms(x, 22, 0x0fc00000)
def CNA_CVT_CON0_CVT_TRUNCATE_2(x): return _ms(x, 16, 0x003f0000)
def CNA_CVT_CON0_CVT_TRUNCATE_1(x): return _ms(x, 10, 0x0000fc00)
def CNA_CVT_CON0_CVT_TRUNCATE_0(x): return _ms(x, 4, 0x000003f0)
def CNA_CVT_CON0_DATA_SIGN(x): return _ms(x, 3, 0x8)
def CNA_CVT_CON0_CVT_TYPE(x): return _ms(x, 1, 0x2)
def CNA_CVT_CON0_CVT_BYPASS(x): return _ms(x, 0, 0x1)
def CNA_CVT_CON1_CVT_SCALE0(x): return _ms(x, 16, 0xffff0000)
def CNA_CVT_CON1_CVT_OFFSET0(x): return _ms(x, 0, 0x0000ffff)
def CNA_CVT_CON2_CVT_SCALE1(x): return _ms(x, 16, 0xffff0000)
def CNA_CVT_CON2_CVT_OFFSET1(x): return _ms(x, 0, 0x0000ffff)
def CNA_CVT_CON3_CVT_SCALE2(x): return _ms(x, 16, 0xffff0000)
def CNA_CVT_CON3_CVT_OFFSET2(x): return _ms(x, 0, 0x0000ffff)
def CNA_CVT_CON4_CVT_SCALE3(x): return _ms(x, 16, 0xffff0000)
def CNA_CVT_CON4_CVT_OFFSET3(x): return _ms(x, 0, 0x0000ffff)
def CNA_CVT_CON5_PER_CHANNEL_CVT_EN(x): return _ms(x, 0, 0xffffffff)
def CNA_PAD_CON0_PAD_LEFT(x): return _ms(x, 4, 0x000000f0)
def CNA_PAD_CON0_PAD_TOP(x): return _ms(x, 0, 0x0000000f)
def CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR(x): return _ms(x, 0, 0xffffffff)
def CNA_DMA_CON0_WEIGHT_BURST_LEN(x): return _ms(x, 16, 0x000f0000)
def CNA_DMA_CON0_DATA_BURST_LEN(x): return _ms(x, 0, 0x0000000f)
def CNA_DMA_CON1_LINE_STRIDE(x): return _ms(x, 0, 0x0fffffff)
def CNA_DMA_CON2_SURF_STRIDE(x): return _ms(x, 0, 0x0fffffff)
def CNA_FC_DATA_SIZE0_DMA_WIDTH(x): return _ms(x, 16, 0x3fff0000)
def CNA_FC_DATA_SIZE0_DMA_HEIGHT(x): return _ms(x, 0, 0x000007ff)
def CNA_FC_DATA_SIZE1_DMA_CHANNEL(x): return _ms(x, 0, 0x0000ffff)
def CNA_PAD_CON1_PAD_VALUE(x): return _ms(x, 0, 0xffffffff)
def CORE_MISC_CFG_QD_EN(x): return _ms(x, 0, 0x1)
def CORE_MISC_CFG_DW_EN(x): return _ms(x, 1, 0x2)
def CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(x): return _ms(x, 16, 0xffff0000)
def CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(x): return _ms(x, 0, 0x0000ffff)
def CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(x): return _ms(x, 0, 0x0000ffff)
def CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(x): return _ms(x, 0, 0x1f)
def DPU_FEATURE_MODE_CFG_BURST_LEN(x): return _ms(x, 5, 0x000001e0)
def DPU_FEATURE_MODE_CFG_OUTPUT_MODE(x): return _ms(x, 1, 0x00000006)
def DPU_FEATURE_MODE_CFG_CONV_MODE(x): return _ms(x, 3, 0x00000018)
def DPU_DATA_CUBE_WIDTH_WIDTH(x): return _ms(x, 0, 0x00001fff)
def DPU_DATA_CUBE_HEIGHT_HEIGHT(x): return _ms(x, 0, 0x00001fff)
def DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(x): return _ms(x, 16, 0x1fff0000)
def DPU_DATA_CUBE_CHANNEL_CHANNEL(x): return _ms(x, 0, 0x00001fff)
def DPU_BS_CFG_BS_ALU_ALGO(x): return _ms(x, 16, 0x000f0000)
def DPU_BS_CFG_BS_ALU_SRC(x): return _ms(x, 8, 0x00000100)
def DPU_BS_CFG_BS_RELU_BYPASS(x): return _ms(x, 6, 0x00000040)
def DPU_BS_CFG_BS_MUL_BYPASS(x): return _ms(x, 4, 0x00000010)
def DPU_BS_OW_CFG_SIZE_E_2(x): return _ms(x, 8, 0x00000700)
def DPU_BS_OW_CFG_SIZE_E_1(x): return _ms(x, 5, 0x000000e0)
def DPU_BS_OW_CFG_SIZE_E_0(x): return _ms(x, 2, 0x0000001c)
def DPU_BS_OW_OP_OW_OP(x): return _ms(x, 0, 0x0000ffff)
def DPU_WDMA_SIZE_0_CHANNEL_WDMA(x): return _ms(x, 0, 0x00001fff)
def DPU_WDMA_SIZE_1_HEIGHT_WDMA(x): return _ms(x, 16, 0x1fff0000)
def DPU_WDMA_SIZE_1_WIDTH_WDMA(x): return _ms(x, 0, 0x00001fff)
def DPU_BN_CFG_BN_RELU_BYPASS(x): return _ms(x, 6, 0x40)
def DPU_BN_CFG_BN_MUL_BYPASS(x): return _ms(x, 4, 0x10)
def DPU_BN_CFG_BN_ALU_BYPASS(x): return _ms(x, 1, 0x2)
def DPU_BN_CFG_BN_BYPASS(x): return _ms(x, 0, 0x1)
def DPU_EW_CFG_EW_CVT_TYPE(x): return _ms(x, 31, 0x80000000)
def DPU_EW_CFG_EW_DATA_MODE(x): return _ms(x, 28, 0x30000000)
def DPU_EW_CFG_EDATA_SIZE(x): return _ms(x, 22, 0x00c00000)
def DPU_EW_CFG_EW_ALU_ALGO(x): return _ms(x, 16, 0x000f0000)
def DPU_EW_CFG_EW_RELU_BYPASS(x): return _ms(x, 9, 0x200)
def DPU_EW_CFG_EW_LUT_BYPASS(x): return _ms(x, 7, 0x80)
def DPU_EW_CFG_EW_OP_SRC(x): return _ms(x, 6, 0x40)
def DPU_EW_CFG_EW_OP_BYPASS(x): return _ms(x, 1, 0x2)
def DPU_EW_CFG_EW_BYPASS(x): return _ms(x, 0, 0x1)
def DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT(x): return _ms(x, 16, 0x003f0000)
def DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(x): return _ms(x, 0, 0x0000ffff)
def DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(x): return _ms(x, 0, 0x0000ffff)
def DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(x): return _ms(x, 0, 0x00000fff)
def DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(x): return _ms(x, 4, 0xfffffff0)
def DPU_SURFACE_ADD_SURF_ADD(x): return _ms(x, 4, 0xfffffff0)
def DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(x): return _ms(x, 0, 0x00001fff)
def DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(x): return _ms(x, 0, 0x00001fff)
def DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(x): return _ms(x, 0, 0x00001fff)
def DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(x): return _ms(x, 1, 0x0000001e)
def DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(x): return _ms(x, 30, 0xc0000000)
def DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(x): return _ms(x, 2, 0x0000000c)
def DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(x): return _ms(x, 0, 0x1)
def DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(x): return _ms(x, 4, 0xfffffff0)
def DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(x): return _ms(x, 11, 0x00007800)
def DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(x): return _ms(x, 8, 0x00000700)
def DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(x): return _ms(x, 4, 0x10)
def DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(x): return _ms(x, 1, 0x6)
def DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(x): return _ms(x, 4, 0xfffffff0)
def DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(x): return _ms(x, 24, 0xff000000)
def DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(x): return _ms(x, 16, 0x00ff0000)
def DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(x): return _ms(x, 8, 0x0000ff00)
def DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(x): return _ms(x, 0, 0x000000ff)
def DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(x): return _ms(x, 4, 0xfffffff0)

def PC_OPERATION_ENABLE_RESERVED_0(x): return _ms(x, 1, 0xfffffffe)
def PC_OPERATION_ENABLE_OP_EN(x): return _ms(x, 0, 0x1)

# **************** Data classes ****************

FEATURE_ATOMIC_SIZE = 16
WEIGHT_ATOMIC_SIZE = 32
ATOMIC_K_SIZE = 16
CBUF_BANK_SIZE = 32768
CBUF_BANKS = 12
CBUF_ENTRIES_PER_BANK = 256
CBUF_ENTRY_SIZE = CBUF_BANK_SIZE // CBUF_ENTRIES_PER_BANK

@dataclass
class SplitTask:
  num:int=0
  top_slice:int=0
  bottom_slice:int=0
  num_overlap_slices:int=0
  num_retain_slices:int=0
  convolutions:int=0
  pad_top:int=0
  pad_bottom:int=0
  pad_left:int=0
  pad_right:int=0
  stride_x:int=1
  stride_y:int=1
  input_width:int=0
  input_height:int=0
  input_channels:int=0
  input_channels_real:int=0
  input_zero_point:int=0
  input_scale:float=1.0
  input_data_entries:int=0
  input_line_stride:int=0
  input_surface_stride:int=0
  input_offset:int=0
  output_width:int=0
  output_height:int=0
  output_channels:int=0
  output_channels_real:int=0
  output_zero_point:int=0
  output_scale:float=1.0
  output_surface_stride:int=0
  output_offset:int=0
  weights_width:int=0
  weights_height:int=0
  weights_kernels:int=0
  weights_zero_point:int=0
  weights_scale:float=1.0
  input_banks:int=0
  weights_banks:int=0
  atomic_count:int=0
  surfaces_per_row:int=0
  regcfg_amount:int=0
  regcfg_addr:int=0

@dataclass
class Operation:
  depthwise:bool=False
  reuse_weights_cbuf:bool=False
  truncate_bits:int=0
  padding_same:bool=True
  stride:int=1
  addition_input:bool=False
  addition_offset:int=0
  addition_scale:float=1.0
  input_index:int=0
  input_width:int=0
  input_height:int=0
  input_channels:int=0
  input_zero_point:int=0
  input_scale:float=1.0
  output_index:int=1
  output_width:int=0
  output_height:int=0
  output_channels:int=0
  output_zero_point:int=0
  output_scale:float=1.0
  weights_width:int=0
  weights_height:int=0
  weights_zero_point:int=0
  weights_scale:float=1.0
  add_tensor:int=-1
  weights_handle:int=0
  biases_handle:int=0
  tasks:List[SplitTask]=field(default_factory=list)

@dataclass
class Plan:
  operations:List[Operation]

# **************** Driver ****************

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

  def submit_jobs(self, jobs:list[drm_rocket_job]):
    jobs_arr = (drm_rocket_job * len(jobs))(*jobs)
    submit = drm_rocket_submit(jobs=ctypes.addressof(jobs_arr), job_count=len(jobs), job_struct_size=ctypes.sizeof(drm_rocket_job), reserved=0)
    self._ioctl(DRM_IOCTL_ROCKET_SUBMIT, submit)

  def synchronize(self): pass
  def __del__(self):
    if hasattr(self, "fd"): os.close(self.fd)

# **************** Tensor packing ****************

def align_up(x:int, a:int) -> int: return ((x + a - 1) // a) * a

def pack_input(input_mv:memoryview, width:int, height:int, channels:int, zero_point:int) -> bytes:
  arr = memoryview(input_mv).cast('B')
  assert len(arr) == width*height*channels
  out = bytearray()
  if channels == 1:
    n = 0
    for x in range(width):
      for y in range(max(height, FEATURE_ATOMIC_SIZE)):
        if y < height: out.append(arr[n]); n += 1
        else: out.append(zero_point)
  else:
    for u in range((channels + FEATURE_ATOMIC_SIZE - 1)//FEATURE_ATOMIC_SIZE):
      for x in range(width):
        for y in range(height):
          for c in range(FEATURE_ATOMIC_SIZE):
            ch = c + u*FEATURE_ATOMIC_SIZE
            if ch < channels: out.append(arr[(y*width + x)*channels + ch]-0x80)
            else: out.append(zero_point-0x80)
  return bytes(out)

def pack_weights(weights_mv:memoryview, wW:int, wH:int, input_channels:int, output_channels:int, zero_point:int, depthwise:bool) -> bytes:
  arr = memoryview(weights_mv).cast('B')
  input_channels_real = input_channels
  output_channels_real = output_channels
  input_channels = max(input_channels, FEATURE_ATOMIC_SIZE)
  output_channels = align_up(output_channels, 2)
  if depthwise: output_channels = 1
  out = bytearray(wW*wH*output_channels*align_up(input_channels, WEIGHT_ATOMIC_SIZE)*2)
  input_channel_groups = WEIGHT_ATOMIC_SIZE * (2 if depthwise else 1)
  input_channels_1 = (input_channels + input_channel_groups - 1)//input_channel_groups
  input_channels_2 = min(input_channels, input_channel_groups)
  n = 0
  for oc1 in range((output_channels + WEIGHT_ATOMIC_SIZE - 1)//WEIGHT_ATOMIC_SIZE):
    for ic1 in range(input_channels_1):
      for x in range(wW):
        for y in range(wH):
          for oc2 in range(min(output_channels, WEIGHT_ATOMIC_SIZE)):
            for ic2 in range(input_channels_2):
              oc = oc1*WEIGHT_ATOMIC_SIZE + oc2
              ic = ic1*input_channel_groups + ic2
              if output_channels_real > 2 and oc >= align_up(output_channels_real, 2):
                continue
              if oc >= output_channels_real:
                out[n] = 0
              elif ic >= input_channels_real:
                if ic2 < 16 or (input_channels_real % 32) > 16:
                  out[n] = (zero_point - 0x80) & 0xff
              else:
                idx = ((oc* wW + x)*wH + y)*input_channels_real + ic
                out[n] = (arr[idx] - 0x80) & 0xff
              n += 1
  return bytes(out[:n])

def calc_bias_correction(weights:memoryview, wW:int, wH:int, ic:int, oc:int, weight_zp:int, input_zp:int, depthwise:bool) -> int:
  corr = 0
  weights_arr = weights.cast('B')
  if depthwise:
    for x in range(wW):
      for y in range(wH):
        corr += (weights_arr[(0*wW + x)*wH*ic + y*ic + oc] - weight_zp)*(input_zp-0x80)
  else:
    for x in range(wW):
      for y in range(wH):
        for c in range(ic):
          corr += (weights_arr[(oc*wW + x)*wH*ic + y*ic + c] - weight_zp)*(input_zp-0x80)
  return corr

def pack_biases(bias_mv:memoryview, weights_mv:memoryview, op:Operation) -> tuple[bytes, int]:
  biases_in = memoryview(bias_mv).cast('i')
  out = bytearray(op.output_channels*4)
  truncate_bits = 0
  for oc in range(op.output_channels):
    corr = calc_bias_correction(weights_mv, op.weights_width, op.weights_height,
                                op.input_channels, oc, op.weights_zero_point,
                                op.input_zero_point, op.depthwise)
    val = (biases_in[oc] - corr) >> truncate_bits
    out[oc*4:(oc+1)*4] = int(val).to_bytes(4, 'little', signed=True)
  return bytes(out), truncate_bits

# **************** Task splitting ****************

def calc_entries_per_slice(op:Operation) -> int:
  atomics_per_entry = CBUF_ENTRY_SIZE // FEATURE_ATOMIC_SIZE
  total_c_atomics = math.ceil(op.input_channels / FEATURE_ATOMIC_SIZE)
  last_c_atomics = total_c_atomics % atomics_per_entry
  int_c_entries = (total_c_atomics // atomics_per_entry) * op.input_width
  frac_c_entries = op.input_width if last_c_atomics == 3 else math.ceil(last_c_atomics * op.input_width / atomics_per_entry)
  return int_c_entries + frac_c_entries

def calc_input_banks(op:Operation) -> int:
  return math.ceil(calc_entries_per_slice(op) * op.input_height / CBUF_ENTRIES_PER_BANK)

def calc_weights_banks(op:Operation) -> int:
  bytes_ = op.weights_width * op.weights_height * op.input_channels
  if not op.depthwise: bytes_ *= op.output_channels
  entries = math.ceil(bytes_ / CBUF_ENTRY_SIZE)
  banks = math.ceil(entries / CBUF_ENTRIES_PER_BANK)
  return banks + 1

def calc_line_stride(width:int) -> int: return width * ATOMIC_K_SIZE

def calc_explicit_padding(op:Operation):
  if op.padding_same and op.weights_width > 1:
    pad_along_width = max((op.output_width - 1)*op.stride + op.weights_width - op.input_width, 0)
    pad_along_height = max((op.output_height - 1)*op.stride + op.weights_height - op.input_height, 0)
    pad_left = pad_along_height // 2
    pad_right = pad_along_height - pad_left
    pad_top = pad_along_width // 2
    pad_bottom = pad_along_width - pad_top
    return pad_top, pad_bottom, pad_left, pad_right
  return 0,0,0,0

def fill_task(op:Operation, task:SplitTask):
  task.stride_x = op.stride
  task.stride_y = op.stride
  task.input_width = op.input_width
  if task.input_width == 8 and (op.addition_input or op.add_tensor != -1): task.input_width *= 2
  task.input_height = op.input_height
  task.input_channels = align_up(max(op.input_channels, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE)
  task.input_channels_real = op.input_channels
  task.input_zero_point = op.input_zero_point
  task.input_scale = op.input_scale
  task.output_width = op.output_width
  task.output_height = op.output_height
  task.output_channels_real = op.output_channels
  task.output_channels = align_up(max(op.output_channels, 32), 32)
  if op.depthwise:
    if task.output_channels_real <= 32: task.output_channels *= 2
    task.output_channels = align_up(task.output_channels, 64)
  task.output_zero_point = op.output_zero_point
  task.output_scale = op.output_scale
  if task.input_channels_real == 1 and (task.output_channels_real > 1 or (op.addition_input or op.add_tensor != -1)):
    task.input_width = max(task.input_width, FEATURE_ATOMIC_SIZE)
    task.input_line_stride = max(calc_line_stride(op.input_width)//FEATURE_ATOMIC_SIZE, FEATURE_ATOMIC_SIZE)
    task.input_surface_stride = int(task.input_line_stride*((task.input_height/4)-1))
  else:
    task.input_line_stride = calc_line_stride(op.input_width)//4
    task.input_surface_stride = int(task.input_line_stride*((task.input_height/4)-1))
  if task.input_width == 8 and (op.addition_input or op.add_tensor != -1):
    task.input_line_stride //= 2
    task.input_surface_stride = 112
  output_line_stride = calc_line_stride(op.output_width)
  task.output_surface_stride = output_line_stride * task.output_height // FEATURE_ATOMIC_SIZE
  if task.input_channels_real == 1:
    task.input_data_entries = task.input_width * task.input_height
  elif task.input_width == 40 and task.input_channels_real == 40:
    task.input_data_entries = 40
  else:
    task.input_data_entries = math.ceil(task.input_width * 2 * math.ceil(task.input_channels_real/FEATURE_ATOMIC_SIZE) / 8)
  task.weights_width = op.weights_width
  task.weights_height = op.weights_height
  task.weights_zero_point = op.weights_zero_point
  task.weights_scale = op.weights_scale
  task.weights_kernels = 1 if op.depthwise else align_up(op.output_channels, 2)
  task.surfaces_per_row = task.output_width * task.output_height * 2
  if op.depthwise: task.surfaces_per_row *= 2

def split_tasks(op:Operation):
  entries_per_slice = calc_entries_per_slice(op)
  input_banks_required = calc_input_banks(op)
  weights_banks_required = calc_weights_banks(op)
  available_weights_banks = weights_banks_required
  available_input_banks = CBUF_BANKS - weights_banks_required
  pad_top, pad_bottom, pad_left, pad_right = calc_explicit_padding(op)
  if weights_banks_required + 1 < CBUF_BANKS:
    op.reuse_weights_cbuf = True
  else:
    op.reuse_weights_cbuf = False
    available_input_banks = 7
    available_weights_banks = CBUF_BANKS - available_input_banks
  if input_banks_required <= available_input_banks:
    t = SplitTask()
    t.num = 0
    fill_task(op, t)
    t.input_banks = input_banks_required
    t.weights_banks = CBUF_BANKS - t.input_banks
    t.input_height = op.input_height
    t.pad_top = pad_top; t.pad_bottom = pad_bottom; t.pad_left = pad_left; t.pad_right = pad_right
    t.atomic_count = t.output_width * t.output_height
    op.tasks.append(t)
    return
  t = SplitTask()
  available_slices = (CBUF_ENTRIES_PER_BANK * available_input_banks) // entries_per_slice
  t.num = 0
  fill_task(op, t)
  t.input_banks = available_input_banks
  t.weights_banks = available_weights_banks
  t.top_slice = 0
  t.bottom_slice = available_slices - 1
  t.pad_top = pad_top; t.pad_left = pad_left; t.pad_right = pad_right
  op.tasks.append(t)
  slice_h = op.weights_height - pad_top - 1
  while slice_h < op.input_height:
    prev = op.tasks[-1]
    while slice_h <= prev.bottom_slice: slice_h += op.stride
    if slice_h > prev.bottom_slice: slice_h -= op.stride
    t = SplitTask()
    t.num = len(op.tasks)
    fill_task(op, t)
    t.top_slice = min(slice_h, prev.bottom_slice) - (op.weights_height - 1) + op.stride
    t.bottom_slice = t.top_slice + available_slices - 1
    t.pad_left = pad_left; t.pad_right = pad_right
    if t.bottom_slice >= op.input_height - 1:
      t.bottom_slice = op.input_height - 1
      t.pad_bottom = pad_bottom
      op.tasks.append(t)
      break
    slice_h = t.top_slice + op.weights_height - 1
    op.tasks.append(t)
  if op.tasks:
    last = op.tasks[-1]
    if last.top_slice >= op.input_height or last.bottom_slice >= (op.input_height + pad_bottom):
      op.tasks.pop()
  for i in range(1, len(op.tasks)):
    prev = op.tasks[i-1]; cur = op.tasks[i]
    if prev.bottom_slice >= cur.top_slice:
      cur.num_overlap_slices = prev.bottom_slice - cur.top_slice + 1
      prev.num_retain_slices = cur.num_overlap_slices
    else:
      cur.num_overlap_slices = 0; prev.num_retain_slices = 0
  output_height_processed = 0
  for i, cur in enumerate(op.tasks):
    slice_h = cur.top_slice + (op.weights_height - 1) - cur.pad_top
    while slice_h <= cur.bottom_slice + cur.pad_bottom:
      slice_h += op.stride
      cur.convolutions += 1
    cur.bottom_slice = min(cur.bottom_slice, op.input_height - 1)
    cur.input_height = cur.bottom_slice - cur.top_slice + 1
    cur.output_width = (cur.input_width + cur.pad_left + cur.pad_right - op.weights_width) // op.stride + 1
    cur.output_height = (cur.input_height + cur.pad_top + cur.pad_bottom - op.weights_height) // op.stride + 1
    cur.atomic_count = cur.output_width * cur.output_height
    cur.input_offset = calc_line_stride(op.input_width) * cur.top_slice
    cur.output_offset = calc_line_stride(op.output_width) * output_height_processed
    cur.input_banks = available_input_banks
    cur.weights_banks = available_weights_banks
    output_height_processed += cur.output_height

# **************** Regcmd emission ****************

def emit_raw(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def emit(reg:int, value:int) -> int:
  return emit_raw(rkt_get_target(reg) + 0x1, reg, value)

def fill_regcmd(op:Operation, task:SplitTask, input_phys:int, weight_phys:int, bias_phys:int, output_phys:int, add_phys:int|None) -> list[int]:
  regs:list[int] = []
  con0 = CNA_CBUF_CON0_WEIGHT_BANK(task.weights_banks) | CNA_CBUF_CON0_DATA_BANK(task.input_banks)
  if task.num > 0 and op.reuse_weights_cbuf: con0 |= CNA_CBUF_CON0_WEIGHT_REUSE(1)
  regs.append(emit(REG_CNA_CBUF_CON0, con0))
  regs.append(emit(REG_CNA_DCOMP_REGNUM, 0))
  regs.append(emit(REG_CNA_DCOMP_CTRL, 0))
  con1 = 0
  if task.input_channels_real == 1:
    con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) | CNA_CONV_CON1_ARGB_IN(8)
  if op.depthwise: con1 |= CNA_CONV_CON1_CONV_MODE(3)
  regs.append(emit(REG_CNA_CONV_CON1, con1))
  regs.append(emit(REG_DPU_RDMA_RDMA_S_POINTER,
                   DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
                   DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
                   DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1)))
  regs.append(emit(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(50 + task.stride_y + 1)))
  regs.append(emit(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task.stride_x) | CNA_CONV_CON3_CONV_Y_STRIDE(task.stride_y)))
  regs.append(emit(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(task.input_width) | CNA_DATA_SIZE0_DATAIN_HEIGHT(task.input_height)))
  regs.append(emit(REG_CNA_DATA_SIZE1,
                   CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(task.input_channels_real - 1) |
                   CNA_DATA_SIZE1_DATAIN_CHANNEL(task.input_channels)))
  regs.append(emit(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(task.output_width)))
  regs.append(emit(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(task.atomic_count)))
  regs.append(emit(REG_CNA_WEIGHT_SIZE0, task.weights_width * task.weights_height * task.input_channels * task.weights_kernels))
  regs.append(emit(REG_CNA_WEIGHT_SIZE1, task.weights_width * task.weights_height * task.input_channels))
  regs.append(emit(REG_CNA_WEIGHT_SIZE2,
                   CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(task.weights_width) |
                   CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(task.weights_height) |
                   CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(task.weights_kernels)))
  regs.append(emit(REG_CNA_CBUF_CON0, con0))
  regs.append(emit(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(task.input_data_entries)))
  if task.input_channels_real == 1:
    truncate = 14; scale = 16384; offset = 65408
    if op.addition_input or op.add_tensor != -1:
      truncate = 15; scale = 32388
    regs.append(emit(REG_CNA_CVT_CON0,
                     CNA_CVT_CON0_CVT_TRUNCATE_3(truncate) |
                     CNA_CVT_CON0_CVT_TRUNCATE_2(truncate) |
                     CNA_CVT_CON0_CVT_TRUNCATE_1(truncate) |
                     CNA_CVT_CON0_CVT_TRUNCATE_0(truncate)))
    regs.append(emit(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(scale) | CNA_CVT_CON1_CVT_OFFSET0(offset)))
    regs.append(emit(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(scale) | CNA_CVT_CON2_CVT_OFFSET1(offset)))
    regs.append(emit(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(scale) | CNA_CVT_CON3_CVT_OFFSET2(offset)))
    regs.append(emit(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(scale) | CNA_CVT_CON4_CVT_OFFSET3(offset)))
  else:
    regs.append(emit(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) | CNA_CVT_CON0_CVT_BYPASS(1)))
    regs.append(emit(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1)))
    regs.append(emit(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1)))
    regs.append(emit(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1)))
    regs.append(emit(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1)))
  regs.append(emit(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(task.pad_left) | CNA_PAD_CON0_PAD_TOP(task.pad_top)))
  regs.append(emit(REG_CNA_FEATURE_DATA_ADDR, CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR(input_phys + task.input_offset)))
  regs.append(emit(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15)))
  regs.append(emit(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task.input_line_stride)))
  regs.append(emit(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task.input_surface_stride)))
  regs.append(emit(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(op.input_width) | CNA_FC_DATA_SIZE0_DMA_HEIGHT(task.input_height)))
  regs.append(emit(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(task.input_channels)))
  regs.append(emit(REG_CNA_DCOMP_CTRL, 0))
  regs.append(emit(REG_CNA_DCOMP_REGNUM, 0))
  regs.append(emit(REG_CNA_DCOMP_ADDR0, weight_phys))
  regs.extend([
    emit(REG_CNA_DCOMP_AMOUNT0, 0), emit(REG_CNA_DCOMP_AMOUNT1, 0), emit(REG_CNA_DCOMP_AMOUNT2, 0),
    emit(REG_CNA_DCOMP_AMOUNT3, 0), emit(REG_CNA_DCOMP_AMOUNT4, 0), emit(REG_CNA_DCOMP_AMOUNT5, 0),
    emit(REG_CNA_DCOMP_AMOUNT6, 0), emit(REG_CNA_DCOMP_AMOUNT7, 0), emit(REG_CNA_DCOMP_AMOUNT8, 0),
    emit(REG_CNA_DCOMP_AMOUNT9, 0), emit(REG_CNA_DCOMP_AMOUNT10, 0), emit(REG_CNA_DCOMP_AMOUNT11, 0),
    emit(REG_CNA_DCOMP_AMOUNT12, 0), emit(REG_CNA_DCOMP_AMOUNT13, 0), emit(REG_CNA_DCOMP_AMOUNT14, 0),
    emit(REG_CNA_DCOMP_AMOUNT15, 0)
  ])
  if task.input_channels_real == 1: regs.append(emit(REG_CNA_CVT_CON5, 65535))
  else: regs.append(emit(REG_CNA_CVT_CON5, 0))
  pad_con1 = (task.input_zero_point - 0x80)
  if task.weights_width >= 3 and task.input_zero_point == 0x0: pad_con1 = 0xffff8080
  if op.addition_input or op.add_tensor != -1: pad_con1 = 0xffffff80
  if op.depthwise and task.input_zero_point == 0x8b: pad_con1 = 0x0b0b
  regs.append(emit(REG_CNA_PAD_CON1, pad_con1 & 0xffffffff))
  misc_cfg = CORE_MISC_CFG_QD_EN(1)
  if op.depthwise: misc_cfg |= CORE_MISC_CFG_DW_EN(1)
  regs.append(emit(REG_CORE_MISC_CFG, misc_cfg))
  regs.append(emit(REG_CORE_DATAOUT_SIZE_0,
                   CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(task.output_height - 1) |
                   CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(task.output_width - 1)))
  regs.append(emit(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(task.output_channels - 1)))
  regs.append(emit(REG_CORE_CLIP_TRUNCATE, CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(op.truncate_bits)))
  regs.append(emit_raw(rkt_get_target(0x3030)+1, 0x3030, 0))
  feat_mode_cfg = DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2)
  if op.depthwise: feat_mode_cfg |= DPU_FEATURE_MODE_CFG_CONV_MODE(3)
  regs.append(emit(REG_DPU_FEATURE_MODE_CFG, feat_mode_cfg))
  regs.append(emit(REG_DPU_DATA_FORMAT, 0))
  regs.append(emit(REG_DPU_OFFSET_PEND, 0))
  regs.append(emit(REG_DPU_DST_BASE_ADDR, output_phys + task.output_offset))
  regs.append(emit(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(task.output_surface_stride)))
  regs.append(emit(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(task.output_width - 1)))
  regs.append(emit(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(task.output_height - 1)))
  regs.append(emit(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0))
  regs.append(emit(REG_DPU_DATA_CUBE_CHANNEL,
                   DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(task.output_channels_real - 1) |
                   DPU_DATA_CUBE_CHANNEL_CHANNEL(task.output_channels - 1)))
  regs.append(emit(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) | DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1)))
  regs.append(emit(REG_DPU_BS_ALU_CFG, 0))
  regs.append(emit(REG_DPU_BS_MUL_CFG, 0))
  regs.append(emit(REG_DPU_BS_RELUX_CMP_VALUE, 0))
  if op.depthwise:
    regs.append(emit(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(3) | DPU_BS_OW_CFG_SIZE_E_1(3) | DPU_BS_OW_CFG_SIZE_E_0(3)))
  else:
    regs.append(emit(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) | DPU_BS_OW_CFG_SIZE_E_0(1)))
  regs.append(emit(REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(0x80 - task.weights_zero_point)))
  regs.append(emit(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(task.output_channels - 1)))
  regs.append(emit(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(task.output_height - 1) | DPU_WDMA_SIZE_1_WIDTH_WDMA(task.output_width - 1)))
  regs.append(emit(REG_DPU_BN_CFG,
                   DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
                   DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1)))
  regs.append(emit(REG_DPU_BN_ALU_CFG, 0))
  regs.append(emit(REG_DPU_BN_MUL_CFG, 0))
  regs.append(emit(REG_DPU_BN_RELUX_CMP_VALUE, 0))
  if op.add_tensor != -1 and add_phys is not None:
    regs.append(emit(REG_DPU_EW_CFG,
                     DPU_EW_CFG_EW_CVT_TYPE(1) | DPU_EW_CFG_EW_DATA_MODE(1) |
                     DPU_EW_CFG_EDATA_SIZE(1) | DPU_EW_CFG_EW_ALU_ALGO(2) |
                     DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) |
                     DPU_EW_CFG_EW_OP_SRC(1)))
    regs.append(emit(REG_DPU_EW_CVT_OFFSET_VALUE, op.addition_offset))
    add_scale = op.addition_scale if op.addition_scale != 0 else 1.0
    scale_bits = math.frexp(add_scale)[0]
    scale_field = max(1 << 14, int(add_scale * (1 << 15)) & 0x7fff)
    regs.append(emit(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT(15) | DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(scale_field)))
    regs.append(emit(REG_DPU_EW_RELUX_CMP_VALUE, 0))
    regs.append(emit(REG_DPU_OUT_CVT_OFFSET, 0))
    regs.append(emit(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale_field)))
    regs.append(emit(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24)))
  else:
    regs.append(emit(REG_DPU_EW_CFG,
                     DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
                     DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
                     DPU_EW_CFG_EW_BYPASS(1)))
    regs.append(emit(REG_DPU_EW_CVT_OFFSET_VALUE, 0))
    regs.append(emit(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1)))
    regs.append(emit(REG_DPU_EW_RELUX_CMP_VALUE, 0))
    conv_scale = (task.input_scale * task.weights_scale) / task.output_scale if task.output_scale != 0 else 1.0
    scale_bits = math.frexp(conv_scale)[0]
    scale_field = max(1 << 14, int(conv_scale * (1 << 15)) & 0x7fff)
    shift = 24
    regs.append(emit(REG_DPU_OUT_CVT_OFFSET, task.output_zero_point - 0x80))
    regs.append(emit(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale_field)))
    regs.append(emit(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift-1)))
  regs.extend([
    emit(REG_DPU_EW_OP_VALUE_0, 0), emit(REG_DPU_EW_OP_VALUE_1, 0), emit(REG_DPU_EW_OP_VALUE_2, 0),
    emit(REG_DPU_EW_OP_VALUE_3, 0), emit(REG_DPU_EW_OP_VALUE_4, 0), emit(REG_DPU_EW_OP_VALUE_5, 0),
    emit(REG_DPU_EW_OP_VALUE_6, 0), emit(REG_DPU_EW_OP_VALUE_7, 0)
  ])
  regs.append(emit(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(task.surfaces_per_row)))
  regs.append(emit_raw(rkt_get_target(0x40c4)+1, 0x40c4, 0))
  regs.extend([emit(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(task.output_width - 1)),
               emit(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(task.output_height - 1)),
               emit(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(task.output_channels - 1))])
  if op.add_tensor != -1 and add_phys is not None:
    regs.append(emit(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, add_phys + task.output_offset))
  else:
    regs.append(emit(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0))
  regs.append(emit(REG_DPU_RDMA_RDMA_BRDMA_CFG, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(1)))
  regs.append(emit(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, bias_phys))
  regs.append(emit(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0))
  regs.append(emit(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0))
  ew_stride = max(op.output_width * op.output_height, 12)
  if op.add_tensor != -1 and add_phys is not None:
    regs.append(emit(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) | DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(1)))
    ew_base_offset = op.output_width * op.output_height * ATOMIC_K_SIZE
    regs.append(emit(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, add_phys + task.output_offset + ew_base_offset))
    regs.append(emit(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(ew_stride)))
  else:
    regs.append(emit(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1)))
    regs.append(emit(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0))
    regs.append(emit(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0))
  rdma_feat_mode_cfg = 0
  if op.add_tensor != -1 and add_phys is not None:
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(5)
  else:
    rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1)
  if op.depthwise: rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(3)
  regs.append(emit(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feat_mode_cfg))
  regs.append(emit(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0))
  surf_notch = ew_stride + task.output_width * (op.output_height - task.output_height)
  if op.input_width == 3: surf_notch = 15
  if op.add_tensor != -1 and add_phys is not None:
    regs.append(emit(REG_DPU_RDMA_RDMA_SURF_NOTCH, DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(surf_notch)))
  else:
    regs.append(emit(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0))
  regs.append(emit(REG_DPU_RDMA_RDMA_PAD_CFG, 0))
  regs.append(emit(REG_DPU_RDMA_RDMA_WEIGHT,
                   DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
                   DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1)))
  if op.add_tensor != -1 and add_phys is not None:
    regs.append(emit(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(surf_notch)))
  else:
    regs.append(emit(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0))
  # Control registers for next regcmd and op enable placeholders
  regs.append(emit(REG_PC_BASE_ADDRESS, 0))
  regs.append(emit(REG_PC_REGISTER_AMOUNTS, 0))
  regs.append(0x0041000000000000)
  regs.append(emit_raw(0x81, REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1)))
  return regs

def stitch_regcmds(regcfgs:list[list[int]], regcmd_phys:int):
  regcmd_bytes = bytearray()
  tasks_info = []
  offset = 0
  for i, regs in enumerate(regcfgs):
    size_words = len(regs)
    padded = (size_words*8 + 63) & ~63
    # patch next
    if i < len(regcfgs)-1:
      next_addr = regcmd_phys + offset + padded
      regs[-4] |= next_addr << 16
      regs[-3] |= (((len(regcfgs[i+1])-4)+1)&~1) << 16
    for w in regs: regcmd_bytes.extend(w.to_bytes(8, 'little'))
    if padded > size_words*8: regcmd_bytes.extend(b"\x00"*(padded-size_words*8))
    tasks_info.append((regcmd_phys + offset, size_words))
    offset += padded
  return bytes(regcmd_bytes), tasks_info

# **************** Compiler / Program ****************

class NPUCompiler(Compiler):
  def __init__(self): super().__init__("compile_npu")
  def compile(self, src:str) -> bytes: return src.encode()

class NPUProgram:
  def __init__(self, dev:'NPUDevice', name:str, lib:bytes):
    self.dev, self.name = dev, name
    self.plan = self._parse_plan(lib)

  def _parse_plan(self, lib:bytes) -> Plan:
    data = json.loads(lib.decode())
    ops = []
    for o in data.get("operations", []):
      ops.append(Operation(
        depthwise=o.get("depthwise", False),
        padding_same=o.get("padding_same", True),
        stride=o.get("stride", 1),
        input_index=o.get("input_index", 0),
        output_index=o.get("output_index", 1),
        weights_width=o["weights_shape"][1],
        weights_height=o["weights_shape"][2],
        input_width=o["input_shape"][1],
        input_height=o["input_shape"][2],
        input_channels=o["input_shape"][3],
        output_width=o["output_shape"][1],
        output_height=o["output_shape"][2],
        output_channels=o["output_shape"][3],
        input_zero_point=o.get("input_zero_point", 0),
        output_zero_point=o.get("output_zero_point", 0),
        weights_zero_point=o.get("weights_zero_point", 0),
        input_scale=o.get("input_scale", 1.0),
        output_scale=o.get("output_scale", 1.0),
        weights_scale=o.get("weights_scale", 1.0),
        add_tensor=o.get("add_tensor", -1),
        addition_scale=o.get("addition_scale", 1.0),
        addition_offset=o.get("addition_offset", 0)
      ))
    return Plan(operations=ops)

  def __del__(self): pass

  def __call__(self, *args, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    # Expect args: output_bo, input_bo, weights_bo, bias_bo, (optional add_bo)
    assert len(args) >= 4, "NPUProgram expects output, input, weights, bias, [add]"
    out_bo, in_bo, weight_bo, bias_bo = args[:4]
    add_bo = args[4] if len(args) > 4 else None
    jobs = []
    for op in self.plan.operations:
      op.weights_handle = weight_bo.handle
      op.biases_handle = bias_bo.handle
      split_tasks(op)
      # pack weights/biases to hw layout
      weight_host = bytearray(weight_bo.size); self.dev.driver.copy_dtoh(weight_host, weight_bo)
      packed_w = pack_weights(weight_host, op.weights_width, op.weights_height,
                              op.input_channels, op.output_channels,
                              op.weights_zero_point, op.depthwise)
      w_bo = self.dev.driver.create_bo(len(packed_w)); self.dev.driver.copy_htod(w_bo, packed_w)
      bias_host = bytearray(bias_bo.size); self.dev.driver.copy_dtoh(bias_host, bias_bo)
      packed_b, trunc = pack_biases(bias_host, weight_host, op)
      op.truncate_bits = trunc
      b_bo = self.dev.driver.create_bo(len(packed_b)); self.dev.driver.copy_htod(b_bo, packed_b)
      regcfgs = []
      for t in op.tasks:
        regcfgs.append(fill_regcmd(op, t, in_bo.dma_addr, w_bo.dma_addr, b_bo.dma_addr, out_bo.dma_addr, add_bo.dma_addr if add_bo else None))
      regcmd_bytes, tasks_info = stitch_regcmds(regcfgs, 0)  # base addr patched later
      reg_bo = self.dev.driver.create_bo(len(regcmd_bytes))
      self.dev.driver.copy_htod(reg_bo, regcmd_bytes)
      # update task addresses with real phys
      tasks = []
      for (addr, count) in tasks_info:
        tasks.append(drm_rocket_task(regcmd=reg_bo.dma_addr + addr, regcmd_count=count))
      tasks_arr = (drm_rocket_task * len(tasks))(*tasks)
      in_handles = (ctypes.c_uint32 * (2 if op.add_tensor != -1 and add_bo else 1))()
      in_handles[0] = in_bo.handle
      if op.add_tensor != -1 and add_bo: in_handles[1] = add_bo.handle
      out_handles = (ctypes.c_uint32 * 1)(out_bo.handle)
      job = drm_rocket_job(tasks=ctypes.addressof(tasks_arr),
                           in_bo_handles=ctypes.addressof(in_handles),
                           out_bo_handles=ctypes.addressof(out_handles),
                           task_count=len(tasks), task_struct_size=ctypes.sizeof(drm_rocket_task),
                           in_bo_handle_count=len(in_handles), out_bo_handle_count=1)
      jobs.append(job)
    self.dev.driver.submit_jobs(jobs)
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
