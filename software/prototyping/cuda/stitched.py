'''
reader --> reorder --> resample
'''

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import scikits.cuda.cufft as cufft

from numpy.fft import irfft,rfft
import numpy as np
from numpy import complex64,float32,float64,int32,uint32,array,arange,empty,zeros,ceil,roll
from struct import unpack

from timing import get_process_cpu_time

import sys
sdbe_scripts_dir = '/home/krosenfe/sdbe/software/prototyping'
sys.path.append(sdbe_scripts_dir)
import read_r2dbe_vdif,read_sdbe_vdif,cross_corr,sdbe_preprocess
import h5py

kernel_template = """
/*
 * GPU kernels for the following pipeline components:
 *   VDIF interpreter
 *   B-engine depacketizer
 *   Pre-preprocessor
 *   Re-ordering
 *   Linear interpolation
 */

#include <cufft.h>

__device__ int32_t get_cid_from_vdif(const int32_t *vdif_start)
{
 return (*(vdif_start + 5) & 0x000000FF);
}

__device__ int32_t get_fid_from_vdif(const int32_t *vdif_start)
{
 return (*(vdif_start + 5) & 0x00FF0000)>>16;
}

__device__ int32_t get_bcount_from_vdif(const int32_t *vdif_start)
{
 return ((*(vdif_start + 5)&0xFF000000)>>24) + ((*(vdif_start + 4)&0x00FFFFFF)<<8);
}

__device__ cufftComplex read_complex_sample(int32_t *samples_int)
{
 float sample_imag, sample_real;
  sample_imag = __int2float_rd(*samples_int & 0x03) - 2.0f;
 *samples_int = (*samples_int) >> 2;
  sample_real = __int2float_rd(*samples_int & 0x03) - 2.0f;
 *samples_int = (*samples_int) >> 2;
 return make_cuFloatComplex(sample_real, sample_imag);
}

__global__ void vdif_to_beng(
 int32_t *vdif_frames,
 int32_t *fid_out,
 int32_t *cid_out,
 int32_t *bcount_out,
 cufftComplex *beng_data_out_0,
 cufftComplex *beng_data_out_1,
 int32_t *beng_frame_completion,
 int32_t num_vdif_frames,
 int32_t bcount_offset)
{

 int32_t cid,fid;
 int32_t bcount;
 const int32_t *vdif_frame_start;
 int32_t samples_per_snapshot_half_0, samples_per_snapshot_half_1;
 int32_t idx_beng_data_out;
 int32_t iframe;
 int idata;
 int isample;
 int old;

 for (iframe=0; iframe + threadIdx.y + blockIdx.x*blockDim.y<num_vdif_frames; iframe+=gridDim.x*gridDim.y*blockDim.y)
 {
# 1000 "reader.cu"
  vdif_frame_start = vdif_frames + (iframe + threadIdx.y + blockIdx.x*blockDim.y)*(1056/4);
# 1015 "reader.cu"
   cid = get_cid_from_vdif(vdif_frame_start);
   fid = get_fid_from_vdif(vdif_frame_start);
   bcount = get_bcount_from_vdif(vdif_frame_start);

  cid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = cid;
  fid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = fid;
  bcount_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = bcount;
# 1040 "reader.cu"
   idx_beng_data_out = 8 * (cid * 8 + fid)*%(BENG_BUFFER_IN_COUNTS)d*128;
   //idx_beng_data_out += ((bcount-bcount_offset)&(%(BENG_BUFFER_IN_COUNTS)d-1))*128;
   idx_beng_data_out += ((bcount-bcount_offset) %% %(BENG_BUFFER_IN_COUNTS)d)*128; // if BENG_BUFFER_IN_COUNTS is not radix-2
# 1058 "reader.cu"
   idx_beng_data_out += threadIdx.x;
# 1112 "reader.cu"
  for (idata=0; idata<(1024/4); idata+=(8/4)*blockDim.x)
  {
# 1124 "reader.cu"
    samples_per_snapshot_half_0 = *(vdif_frame_start + (32/4) + idata + (8/4)*threadIdx.x);
    samples_per_snapshot_half_1 = *(vdif_frame_start + (32/4) + idata + (8/4)*threadIdx.x + 1);

   for (isample=0; isample<8/2; isample++)
   {

     beng_data_out_1[idx_beng_data_out+(8/2-(isample+1))*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_0);
     beng_data_out_0[idx_beng_data_out+(8/2-(isample+1))*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_0);
     beng_data_out_1[idx_beng_data_out+(8/2-(isample+1)+8/2)*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_1);
     beng_data_out_0[idx_beng_data_out+(8/2-(isample+1)+8/2)*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_1);
# 1193 "reader.cu"
   }
    idx_beng_data_out += blockDim.x;
  }
# 1253 "reader.cu"
  old = atomicAdd(beng_frame_completion + ((bcount-bcount_offset) %% %(BENG_BUFFER_IN_COUNTS)d), 1);
# 1277 "reader.cu"
  if (__any(old == ((16384/8)*blockDim.x)-1))
  {
# 1298 "reader.cu"
  }
 }
}


__global__ void reorderTz_smem(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){
  // gridDim.x = 16384 * 128 / (16 * 16) = 8192
  // blockDim.x = 16; blockDim.y = 16;
  // --> launches 2097152 threads

  int32_t sid_out,bid_in;

  __shared__ cufftComplex tile[16][16];

  // for now, let us loop the grid over B-engine frames:
  for (int bid_out=0; bid_out<num_beng_frames-1; bid_out+=1){

    // input snapshot id
    int sid_in = (blockIdx.x * blockDim.x + threadIdx.x) %% 128;
    // input channel id 
    int cid = threadIdx.y + blockDim.y * (blockIdx.x / (128 / blockDim.x));

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    if (sid_out < 69){
      bid_in = bid_out;
    } else {
      bid_in = bid_out+1;
    }

    tile[threadIdx.x][threadIdx.y] = beng_data_in[128*num_beng_frames*cid + 128*bid_in + sid_in];

    __syncthreads();

    // now we tranpose warp orientation over channels and snapshot index

    // snapshot id
    sid_in = threadIdx.y + (blockIdx.x*blockDim.y) %% 128;
    // channel id 
    cid = threadIdx.x + blockDim.x * (blockIdx.x / (128 / blockDim.x)); 

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    beng_data_out[128*16385*bid_out + 16385*sid_out + cid] = tile[threadIdx.y][threadIdx.x];

    // zero out nyquist: 
    if (cid == 0) {
      beng_data_out[128*16385*bid_out + 16385*sid_out + 16384] = make_cuComplex(0.,0.);
    }

    __syncthreads();
  }
}

__global__ void linear(float *a, int Na, float *b, int Nb, double c, float d){
 /*
  This kernel uses a round-half-to-even tie-breaking rule which is
  opposite that of python's interp_1d.
  deal with extrapolation
  a: input_array (assume padded by two floats for every SWARM snapshot)
  b: output_array
  Nb: size of array b
  c: conversion factor between a and b indices. 
  Note: type conversions are slowing this down.
  Idea: Use texture memory to store ida and weights.
  */

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < Nb) {
    int ida = __double2int_rd(tid*c); // round down
    if (ida < Na-1){
      b[tid] = d * ( a[(ida / 32768)*32770 + (ida %% 32768) ]*(1.-(c*tid-ida)) + 
		     a[((ida+1) / 32768)*32770 + ((ida+1) %% 32768)]*(c*tid-ida) );
    } else {
      b[tid] = d * a[(ida / 32768)*32770 + (ida %% 32768) ];
    }
  }
}

__global__ void zero_out(cufftComplex *a, int32_t n)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n){
    a[tid] = make_cuComplex(0.,0.);
  }
}

__global__ void strided_copy(float *a, int istart, float *b, int N, int istride, int iskip)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < N){
    b[tid] = a[istart+(tid/istride)*(istride+iskip) + (tid %% istride) ];
  }
}
"""


def get_bcount_from_vdif(vdif_start):
  '''
  read .vdif file into uint32 word array 
  '''
  BENG_VDIF_HDR_0_OFFSET_INT = 4  # b1 b2 b3 b4
  BENG_VDIF_HDR_1_OFFSET_INT = 5  #  c  0  f b0
  return ((vdif_start[BENG_VDIF_HDR_1_OFFSET_INT]&0xFF000000)>>24) + ((vdif_start[BENG_VDIF_HDR_0_OFFSET_INT]&0x00FFFFFF)<<8);

def read_vdif(filename_input,num_vdif_frames,beng_frame_offset=1,batched=True):
  vdif_buf = empty(VDIF_BYTE_SIZE*num_vdif_frames/4, dtype=uint32)
  for ii in arange(4):
    filename_data = "%s_eth%d.vdif" % (filename_input, 2+ii)
    print '\n',filename_data
    fh = open(filename_data,"r")
    # seek to specified B-engine frame boundary
    tmp_vdif = array(unpack('<{0}I'.format(VDIF_INT_SIZE),fh.read(VDIF_BYTE_SIZE)), uint32)
    tmp_bcount_curr = get_bcount_from_vdif(tmp_vdif)
    tmp_bcount_prev = tmp_bcount_curr
    while (tmp_bcount_curr-tmp_bcount_prev < beng_frame_offset):
      tmp_vdif = array(unpack('<{0}I'.format(VDIF_INT_SIZE),fh.read(VDIF_BYTE_SIZE)), uint32)
      tmp_bcount_curr = get_bcount_from_vdif(tmp_vdif)
    print 'B-count = %d = %d, seeking one frame back' % (tmp_bcount_curr, tmp_bcount_prev+beng_frame_offset)
    fh.seek(-1*VDIF_BYTE_SIZE,1)
    bcount_offset = tmp_bcount_curr
    # read data
    vdif_buf[ii*VDIF_INT_SIZE*num_vdif_frames/4:(ii+1)*VDIF_INT_SIZE*num_vdif_frames/4] = array(unpack('<{0}I'.format(VDIF_INT_SIZE*num_vdif_frames/4),fh.read(VDIF_BYTE_SIZE * num_vdif_frames / 4)), uint32)
    fh.seek(-1*VDIF_BYTE_SIZE,1)
    tmp_vdif = array(unpack('<{0}I'.format(VDIF_INT_SIZE),fh.read(VDIF_BYTE_SIZE)), uint32)
    print 'ending with B-count = %d, read %d B-engine frames' % (get_bcount_from_vdif(tmp_vdif), get_bcount_from_vdif(tmp_vdif)-bcount_offset+1)
    fh.close()
  return vdif_buf,bcount_offset

def meminfo(kernel):
  print "\nRegisters: %d" % kernel.num_regs
  print "Local: %d" % kernel.local_size_bytes
  print "Shared: %d" % kernel.shared_size_bytes
  print "Const: %d" % kernel.const_size_bytes

def gpumeminfo(driver):
  print 'Memory usage:', 1.- 1.*driver.mem_get_info()[0]/driver.mem_get_info()[1]

##########################################################################
##########################################################################

VDIF_BYTE_SIZE = 1056
VDIF_BYTE_SIZE_DATA = 1024
VDIF_INT_SIZE = VDIF_BYTE_SIZE/4
VDIF_PER_BENG = 2048

BENG_CHANNELS_ = 16384
BENG_CHANNELS = (BENG_CHANNELS_ + 1)
BENG_SNAPSHOTS = 128

SWARM_RATE = 2496e6
R2DBE_RATE = 4096e6

BENG_BUFFER_IN_COUNTS = 40
SNAPSHOTS_PER_BATCH = 39

# settings
beng_frame_offset = 1
scan_filename_base = 'prep6_test1_local'
filename_input = '/home/shared/sdbe_preprocessed/'+scan_filename_base+'_swarmdbe'
DEBUG = False
#interp_kind = 'linear'
interp_kind = 'fft'

# derive quantities
num_vdif_frames = BENG_BUFFER_IN_COUNTS*VDIF_PER_BENG
num_swarm_samples = int((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*2*BENG_CHANNELS_)
num_r2dbe_samples = int(num_swarm_samples*R2DBE_RATE/SWARM_RATE)

# compile CUDA kernels
kernel_source = kernel_template % {'BENG_BUFFER_IN_COUNTS':BENG_BUFFER_IN_COUNTS}
kernel_module = SourceModule(kernel_source)

# fetch kernel handles
vdif_to_beng_kernel = kernel_module.get_function('vdif_to_beng')
reorderTz_smem_kernel = kernel_module.get_function('reorderTz_smem')
linear_kernel = kernel_module.get_function('linear')
zero_out_kernel = kernel_module.get_function('zero_out')
strided_copy_kernel = kernel_module.get_function('strided_copy')

# read vdif
cpu_vdif_buf,bcount_offset = read_vdif(filename_input,num_vdif_frames,beng_frame_offset,batched=True)

# inverse in-place FFT plan
n = array([2*BENG_CHANNELS_],int32)
inembed = array([BENG_CHANNELS],int32)
onembed = array([2*BENG_CHANNELS],int32)
plan_A = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, BENG_CHANNELS,
	                                       onembed.ctypes.data, 1, 2*BENG_CHANNELS,
  					       cufft.CUFFT_C2R, (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS)

# band trimming R2C FFT plan
n = array([4096],int32)
inembed = array([4096],int32)
onembed = array([4096/2+1],int32)
batch_B = num_r2dbe_samples / 4096 / 128
plan_B = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, 4096,
					       onembed.ctypes.data, 1, 4096/2+1,
					       cufft.CUFFT_R2C, batch_B)

# band trimming C2R FFT plan
n = array([2048],int32)
inembed = array([2048/2+1],int32)
onembed = array([2048],int32)
batch_C = batch_B
plan_C = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, 4096/2+1,
					       onembed.ctypes.data, 1, 2048,
					       cufft.CUFFT_C2R, batch_C)

if interp_kind == 'fft':
  # Turn concatenated SWARM time series into single spectrum.
  # Note that the input is padded with two extra values from plan_A.
  n = array([39*2*BENG_CHANNELS_],int32)
  inembed = array([39*2*BENG_CHANNELS_],int32)
  onembed = array([39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1],int32)
  plan_interp_A = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,39*2*BENG_CHANNELS_,
					onembed.ctypes.data,1,int(39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1),
					cufft.CUFFT_R2C,1)
  # Turn padded SWARM spectrum into time series with R2DBE sampling rate
  n = array([39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE],int32)
  inembed = array([39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1],int32)
  onembed = array([39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE],int32)
  plan_interp_B = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,int(39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1),
					onembed.ctypes.data,1,int(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE),
					cufft.CUFFT_C2R,1)
  
# event timers
tic = cuda.Event()
toc = cuda.Event()

# allocate device memory
# reader
gpu_vdif_buf    = cuda.mem_alloc(cpu_vdif_buf.nbytes)
gpu_beng_data_0 = cuda.mem_alloc(8*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS)
gpu_beng_data_1 = cuda.mem_alloc(8*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS)
gpu_fid         = cuda.mem_alloc(4*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS)
gpu_cid         = cuda.mem_alloc(4*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS)
gpu_bcount      = cuda.mem_alloc(4*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS)
gpu_beng_frame_completion = cuda.mem_alloc(4*BENG_BUFFER_IN_COUNTS)
# reorder/resample
# buffers

tick = get_process_cpu_time()

# move cpu_vdif_buf to device
tic.record()
cuda.memcpy_htod(gpu_vdif_buf,cpu_vdif_buf)
toc.record()
toc.synchronize()
print 'htod transfer rate: %g GB/s' % (cpu_vdif_buf.nbytes/(tic.time_till(toc)*1e-3),)

# depacketize and dequantize BENG_BUFFER_IN_COUNTS
# gpu_beng_data_0 is USB
tic.record()
blocks_per_grid = 128
vdif_to_beng_kernel(	gpu_vdif_buf,
			gpu_fid, 
			gpu_cid, 
			gpu_bcount,
			gpu_beng_data_0,
			gpu_beng_data_1,
			gpu_beng_frame_completion,
			int32(BENG_BUFFER_IN_COUNTS*VDIF_PER_BENG),
			int32(bcount_offset),
  			block=(32,32,1), grid=(blocks_per_grid,1,1))
if DEBUG:
  print 'DEBUG::loading cpu_beng_data_1'
  gpumeminfo(cuda)
  cpu_beng_data_1 = empty((BENG_CHANNELS_,BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS),dtype=complex64)
  cuda.memcpy_dtoh(cpu_beng_data_1,gpu_beng_data_1)

# reorder and zero pad SWARM snapshots
gpu_beng_0 = cuda.mem_alloc(8*BENG_CHANNELS*BENG_SNAPSHOTS*(BENG_BUFFER_IN_COUNTS-1))
gpu_beng_1 = cuda.mem_alloc(8*BENG_CHANNELS*BENG_SNAPSHOTS*(BENG_BUFFER_IN_COUNTS-1))
reorderTz_smem_kernel(gpu_beng_data_1,gpu_beng_1,np.int32(BENG_BUFFER_IN_COUNTS),
	block=(16,16,1),grid=(BENG_CHANNELS_*BENG_SNAPSHOTS/(16*16),1,1),)
# free unpacked, unordered b-frames
gpu_beng_data_0.free()
gpu_beng_data_1.free()

if DEBUG:
  print 'DEBUG::loading cpu_beng_spectra_1'
  gpumeminfo(cuda)
  cpu_beng_spectra_1 = empty(((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS,BENG_CHANNELS),dtype=complex64)
  cuda.memcpy_dtoh(cpu_beng_spectra_1,gpu_beng_1)

# allocate more memory
gpu_r2dbe = cuda.mem_alloc(4 * num_r2dbe_samples)
gpu_r2dbe_spec = cuda.mem_alloc(8 * (4096/2+1) * batch_B)
gpu_r2dbe_trimmed = cuda.mem_alloc(4*num_r2dbe_samples/4096*2048)

for SB in (gpu_beng_0,gpu_beng_1):
#for SB in (gpu_beng_1,):
  # Turn SWARM snapshots into timeseries
  cufft.cufftExecC2R(plan_A,int(SB),int(SB))
  # gpu_beng_[0-1] are now padded by two floats every 2*BENG_CHANNELS_ samples: [num_snapshots, 2*16384 + 2]
  if DEBUG:
    print 'DEBUG::loading cpu_beng_timeseries_1'
    gpumeminfo(cuda)
    cpu_beng_timeseries_1 = empty(((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS,2*BENG_CHANNELS),dtype=float32)
    cuda.memcpy_dtoh(cpu_beng_timeseries_1,gpu_beng_1)
    cpu_beng_timeseries_1 = cpu_beng_timeseries_1[:,:2*BENG_CHANNELS_]

  if interp_kind == 'linear':
    # Resample the entire time series using linear interpolation and rescale.
    threads_per_block = 512
    blocks_per_grid = int(ceil(1. * num_r2dbe_samples / threads_per_block))
    linear_kernel(	SB,
		int32(num_swarm_samples),
	      	gpu_r2dbe,
		int32(num_r2dbe_samples),
		float64(SWARM_RATE/R2DBE_RATE),
		float32(1./(2*BENG_CHANNELS_)),
		block=(threads_per_block,1,1),grid=(blocks_per_grid,1))
    #gpu_beng_0.free() # can't do this yet!
    SB.free()
  elif interp_kind == 'fft':
    # look over batch=39 snapshots
    gpu_tmp = cuda.mem_alloc(8*int(39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1))
    gpu_bar = cuda.mem_alloc(4*int(39*2*BENG_CHANNELS_))
    # zero out gpu_tmp
    zero_out_kernel(gpu_tmp,int32(39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1),
			block=(1024,1,1),grid=(int(ceil((39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1)/1024.)),1))
    cpu_tmp = zeros(39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1,dtype=complex64)
    #cuda.memcpy_dtoh(cpu_tmp,gpu_tmp)
    #print 'check zero out:',cpu_tmp.max(), cpu_tmp.min()
    for ib in range((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS/39):
      # copy 1 SWARM time sereis chunk, removing padding
      strided_copy_kernel(SB,int32(39*2*BENG_CHANNELS*ib),gpu_bar,
			int32(39*2*BENG_CHANNELS_),int32(2*BENG_CHANNELS_),int32(2),
		block=(512,1,1),grid=(39*2*BENG_CHANNELS_/512,1))
      # Turn concatenated SWARM time series into single spectrum (already zero padded)
      #cufft.cufftExecR2C(plan_interp_A,int(SB)+int(4*39*2*BENG_CHANNELS*ib),int(gpu_tmp))
      cufft.cufftExecR2C(plan_interp_A,int(gpu_bar),int(gpu_tmp))
      cuda.memcpy_dtoh(cpu_tmp,gpu_tmp)
      # Turn padded SWARM spectrum into time series with R2DBE sampling rate
      cufft.cufftExecC2R(plan_interp_B,
			int(gpu_tmp),
			int(gpu_r2dbe)+int(4*39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE*ib))

      # note that we need to normalize: 39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE*(2*BENG_CHANNELS_)

    gpu_tmp.free()
    gpu_bar.free()
    SB.free()

  if DEBUG:
    print 'DEBUG::loading resampled time series cpu_r2dbe'
    gpumeminfo(cuda)
    cpu_r2dbe = np.empty(num_r2dbe_samples,dtype=float32)
    cuda.memcpy_dtoh(cpu_r2dbe,gpu_r2dbe)

  # loop through resampled time series in chunks of batch_B num_r2dbe_samples/4096/batch_B
  for ib in range(num_r2dbe_samples/4096/batch_B):

    # compute spectrum with Fs = 4096
    cufft.cufftExecR2C(plan_B,int(gpu_r2dbe)+int(4*ib*4096*batch_B),int(gpu_r2dbe_spec))

    # invert to time series with B=1024, masking out first 150 MHz and last (1024 - 150) MHz. (MUST INDEX BY 8...)
    cufft.cufftExecC2R(plan_C,int(gpu_r2dbe_spec)+int(8*150),int(gpu_r2dbe_trimmed) + int(4*ib*2048*batch_B))

  if DEBUG:
    print 'DEBUG::loading batch of low res, resampled spectra cpu_r2dbe_spec'
    # grab trimmed timeseries
    gpumeminfo(cuda)
    cpu_r2dbe_spec = np.empty((batch_B,4096/2+1),dtype=complex64)
    cuda.memcpy_dtoh(cpu_r2dbe_spec,gpu_r2dbe_spec)

    #plt.plot(np.mean(np.abs(cpu_r2dbe_spec[:,150:-874]),axis=0))


toc.record()
toc.synchronize()
tock = get_process_cpu_time()

time_gpu = tic.time_till(toc)
time_cpu = tock - tick

print ''
print 'time resampled:', 13.128e-3 * BENG_SNAPSHOTS * (BENG_BUFFER_IN_COUNTS - 1), ' ms'
print 'Transfer size was %d bytes' % cpu_vdif_buf.nbytes
print 'GPU time:',time_gpu,' ms'
print 'CPU:',time_cpu.nanoseconds*1e-6,' ms'

## DEBUGGING:
# grab resampled and trimmed time series
cpu_r2dbe_trimmed = np.empty(2048*num_r2dbe_samples/4096,float32)
cuda.memcpy_dtoh(cpu_r2dbe_trimmed,gpu_r2dbe_trimmed)
# (should be divided by 2048 to retain same scale)

## Normalize:
if DEBUG:
  cpu_r2dbe /= 39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE*(2*BENG_CHANNELS_)
if interp_kind is 'linear':
  cpu_r2dbe_trimmed /= 2048.
elif interp_kind is 'fft':
  cpu_r2dbe_trimmed /= 2048.*39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE*(2*BENG_CHANNELS_)

if DEBUG:
  import matplotlib.pyplot as plt

# destroy plans
cufft.cufftDestroy(plan_A)
cufft.cufftDestroy(plan_B)
cufft.cufftDestroy(plan_C)

# free memory
gpu_r2dbe_spec.free()
gpu_r2dbe.free()
gpu_r2dbe_trimmed.free()

#if DEBUG:
if False:
  # Now read R2DBE data covering roughly the same time window as the SWARM
  # data. Start at an offset of zero (i.e. from the first VDIF packet) to
  # keep things simple.
  N_r_vdif_frames = int(np.ceil(read_sdbe_vdif.SWARM_TRANSPOSE_SIZE*(BENG_BUFFER_IN_COUNTS-1)*read_sdbe_vdif.R2DBE_RATE/read_sdbe_vdif.SWARM_RATE))
  vdif_frames_offset = 0
  rel_path_to_in = '/home/shared/sdbe_preprocessed/'
  d = sdbe_preprocess.get_diagnostics_from_file(scan_filename_base,rel_path=rel_path_to_in)
  xr = read_r2dbe_vdif.read_from_file(rel_path_to_in + scan_filename_base + '_r2dbe_eth3.vdif',N_r_vdif_frames,vdif_frames_offset)

  # compare to pre-shifted data
  hf5 = h5py.File('prep6_test1_local_sdbe_preprocess.hdf5')
  spectra = hf5.get('Xs1').value
  #spectra = np.roll(cpu_beng_spectra_1,-d['get_idx_offset'][0],axis=0)
  xs_shifted = sdbe_preprocess.resample_sdbe_to_r2dbe_fft_interp(spectra,interp_kind="linear")

  # Do FX correlation search on the two time-domain signals BEFORE band trimming
  s_range = arange(-16,16)
  s_avg = 128
  offset_swarmdbe_data = d['offset_swarmdbe_data']
  idx_offset = d['get_idx_offset'][0]
  fft_window_size = 32768*2
  s_0x1_shifted, S_0x1_shifted, s_peaks_shifted = cross_corr.corr_FXt(xr,xs_shifted[offset_swarmdbe_data:],
									fft_window_size=fft_window_size,search_range=s_range,search_avg=s_avg)
  s_0x1, S_0x1, s_peaks = cross_corr.corr_FXt(xr,cpu_r2dbe[np.ceil(idx_offset*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)+offset_swarmdbe_data:],
							fft_window_size=fft_window_size,search_range=s_range,search_avg=s_avg)  
  print 'max peak:',s_peaks.max()

  # foo: 0.179,
  # linear: 0.18216315
  # fft: 0.17923266
  plt.stem(s_range,s_peaks_shifted,markerfmt='b^')
  plt.stem(s_range,s_peaks)
  plt.xlabel('FFT window offset')
  plt.ylabel('Corr coef (peak per window)')


  # now we check the band limited series
  #xr_bl = sdbe_preprocess.bandlimit_1248_to_1024(xr,sub_sample=True)
  #xs_bl = sdbe_preprocess.bandlimit_1248_to_1024(xs_shifted[:xs_shifted.size/4096 * 4096],sub_sample=True)
  ##s_bl_shifted_0x1, S_bl_shifted_0x1, s_bl_shifted_peaks = cross_corr.corr_FXt(xr_bl,xs_bl[np.ceil(idx_offset*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)+offset_swarmdbe_data/2:],
  #s_bl_shifted_0x1, S_bl_shifted_0x1, s_bl_shifted_peaks = cross_corr.corr_FXt(xr_bl,xs_bl[offset_swarmdbe_data/2:],
  #							fft_window_size=fft_window_size,search_range=s_range,search_avg=s_avg)  
  #s_bl_0x1, S_bl_0x1, s_bl_peaks = cross_corr.corr_FXt(xr_bl,cpu_r2dbe_trimmed[np.ceil(idx_offset*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)+offset_swarmdbe_data/2:],
  #							fft_window_size=fft_window_size,search_range=s_range,search_avg=s_avg)  

  # 0.06, problem is shifting....
  # 0.32661165,0.06

  #plt.figure()
  #plt.stem(s_range,s_bl_shifted_peaks,markerfmt='^')
  #plt.stem(s_range,s_bl_peaks)
  #plt.xlabel('FFT window offset')
  #plt.ylabel('Corr coef (peak per window)')

  plt.ion()
  plt.show()

tmp = rfft(cpu_beng_timeseries_1.reshape((128,39*2*BENG_CHANNELS_)),axis=-1)
bar = np.hstack([ tmp,zeros((128,39*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE -39*BENG_CHANNELS_)) ])
foo = irfft(bar,axis=-1)
foo /= (2*BENG_CHANNELS_)

# timing at ~4.3 x real!
print 'done!'
