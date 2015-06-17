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
import logging

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


__global__ void strided_copy(float *a, int istart, float *b, int N, int istride, int iskip)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < N){
    b[tid] = a[istart + (tid/istride)*(istride+iskip) + (tid %% istride) ];
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

def corr_FXt(x0,x1,fft_window_size=32768,search_range=None,search_avg=1):
	"""
	Do FX cross-correlation by subdividing time-series into FFT windows.
	
	Optionally perform a search over multiple FFT window offsets.

	Arguments:
	----------
	x0,x1 -- Time-domain signals.
	fft_window_size -- The number of samples to take in an FFT window.
	search_range -- Range of FFT window offsets to search for cross-
	correlation, or None to not do search (default is None).
	search_avg -- Number of windows over which to average when doing a
	search. If search is not performed this parameter has no impact
	(default is 1).
	
	Returns:
	--------
	s_0x1 -- Time-domain cross-correlation of two signals. If search is
	done this is two-dimensional, with relative window offset along the
	zeroth dimension.
	S_0x1 -- Cross-power spectrum of two signals. If search is done
	this is two-dimensional, with relative window offset along the 
	zeroth dimension.
	s_peaks -- If search is done, this returns the peak in the cross-
	correlation as a function of relative window offset.
	
	Notes:
	------
	The search is done from the center window in x0, and over the search
	range, relative to that window, in x1.
	Code written by cross_corr.py by Andre Young.

	Changes:
	------
	Uses rfft and assumes real Time-domain signal
	Removes phase
	"""
	
	N_samples = min((2**int(floor(log2(x0.size))),2**int(floor(log2(x1.size)))))
	X0 = rfft(x0[:N_samples].reshape((N_samples/fft_window_size,fft_window_size)),axis=1)
	X1 = rfft(x1[:N_samples].reshape((N_samples/fft_window_size,fft_window_size)),axis=1)
	
	if (search_range == None):
		s_peaks = None
		s_0x1,S_0x1 = corr_Xt(X0,X1,fft_window_size=fft_window_size)
	else:
		# do search
		s_0x1,S_0x1,s_peaks = corr_Xt_search(X0,X1,fft_window_size=fft_window_size,search_range=search_range,search_avg=search_avg)
	
	return s_0x1,S_0x1,s_peaks

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
DEBUG = True 
in_place = False

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
strided_copy_kernel = kernel_module.get_function('strided_copy')

# read vdif
cpu_vdif_buf,bcount_offset = read_vdif(filename_input,num_vdif_frames,beng_frame_offset,batched=True)

# inverse in-place FFT plan
if in_place:
  n = array([2*BENG_CHANNELS_],int32)
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS],int32)
  plan_A = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, BENG_CHANNELS,
	                                       onembed.ctypes.data, 1, 2*BENG_CHANNELS,
  					       cufft.CUFFT_C2R, (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS)
else:
  # inverse FFT plan
  n = array([2*BENG_CHANNELS_],int32)
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  plan_A = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, BENG_CHANNELS,
	                                       onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
	  				       cufft.CUFFT_C2R, (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS)

# Turn concatenated SWARM time series into single spectrum
n = array([39*2*BENG_CHANNELS_],int32)
inembed = array([39*2*BENG_CHANNELS_],int32)
onembed = array([39*BENG_CHANNELS_+1],int32)
plan_interp_A = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,39*2*BENG_CHANNELS_,
					onembed.ctypes.data,1,39*BENG_CHANNELS_+1,
					cufft.CUFFT_R2C,1)


# Turn trimmed spectrum into 2048 timeseries
n = array([32*2*BENG_CHANNELS_],int32)
inembed = array([39*BENG_CHANNELS_+1],int32)
onembed = array([32*2*BENG_CHANNELS_],int32)
plan_interp_B = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,39*BENG_CHANNELS_+1,
					onembed.ctypes.data,1,32*2*BENG_CHANNELS_,
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

# allocate memory for time series
gpu_r2dbe = cuda.mem_alloc(4 * num_r2dbe_samples / 2)
if not in_place: 
  gpu_swarm = cuda.mem_alloc(4 * num_swarm_samples)

for SB in (gpu_beng_0,gpu_beng_1):
#for SB in (gpu_beng_1,):
  # Turn SWARM snapshots into timeseries
  if in_place:
    cufft.cufftExecC2R(plan_A,int(SB),int(SB))
  else:  
    cufft.cufftExecC2R(plan_A,int(SB),int(gpu_swarm))
    SB.free()
  if DEBUG:
    print 'DEBUG::loading cpu_beng_timeseries_1'
    gpumeminfo(cuda)
    if in_place:
      cpu_beng_timeseries_1 = empty(((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS,2*BENG_CHANNELS),dtype=float32)
      cuda.memcpy_dtoh(cpu_beng_timeseries_1,SB)
      cpu_beng_timeseries_1 = cpu_beng_timeseries_1[:,:2*BENG_CHANNELS_]
    else:
      cpu_beng_timeseries_1 = empty(((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS,2*BENG_CHANNELS_),dtype=float32)
      cuda.memcpy_dtoh(cpu_beng_timeseries_1,gpu_swarm)

  # look over chunks of 39 SWARM snapshots
  gpu_tmp = cuda.mem_alloc(8*int(39*BENG_CHANNELS_+1))
  if in_place:
    gpu_bar = cuda.mem_alloc(4*int(39*2*BENG_CHANNELS_))
  for ib in range((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS/39):

    if in_place:
      # copy 39 snapshots into a signal time series chunk, removing padding
      strided_copy_kernel(SB,int32(39*2*BENG_CHANNELS*ib),gpu_bar,
			int32(39*2*BENG_CHANNELS_),int32(2*BENG_CHANNELS_),int32(2),
			block=(512,1,1),grid=(39*2*BENG_CHANNELS_/512,1))
      # Turn concatenated SWARM time series into single spectrum
      cufft.cufftExecR2C(plan_interp_A,int(gpu_bar),int(gpu_tmp))

    else:
      # Turn concatenated SWARM time series into single spectrum
      cufft.cufftExecR2C(plan_interp_A,int(gpu_swarm)+int(4*39*2*BENG_CHANNELS_*ib),int(gpu_tmp))

    # Turn padded SWARM spectrum into time series with R2DBE sampling rate
    cufft.cufftExecC2R(plan_interp_B,
		int(gpu_tmp)+int(8*150*512),int(gpu_r2dbe)+int(4*32*2*BENG_CHANNELS_*ib))
			#int(gpu_tmp)+int(8*(1024-150)*512),int(gpu_r2dbe)+int(4*32*2*BENG_CHANNELS_*ib))

      # note that we need to normalize: 39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE*(2*BENG_CHANNELS_)

  if in_place:
    gpu_bar.free()
    SB.free()
  gpu_tmp.free()

if not in_place:
  gpu_swarm.free()

toc.record()
toc.synchronize()

time_gpu = tic.time_till(toc)

print 'DEBUG::loading resampled time series cpu_r2dbe'
gpumeminfo(cuda)
cpu_r2dbe = np.empty(num_r2dbe_samples/2,dtype=float32)
cuda.memcpy_dtoh(cpu_r2dbe,gpu_r2dbe)

print ''
print 'time resampled:', 13.128e-3 * BENG_SNAPSHOTS * (BENG_BUFFER_IN_COUNTS - 1), ' ms'
print 'Transfer size was %d bytes' % cpu_vdif_buf.nbytes
print 'GPU time:',time_gpu,' ms'

# destroy plans
cufft.cufftDestroy(plan_A)
cufft.cufftDestroy(plan_interp_A)
cufft.cufftDestroy(plan_interp_B)

# free memory
#gpu_r2dbe.free()

if DEBUG:
  import matplotlib.pyplot as plt
  # Now read R2DBE data covering roughly the same time window as the SWARM
  # data. Start at an offset of zero (i.e. from the first VDIF packet) to
  # keep things simple.
  N_r_vdif_frames = int(np.ceil(read_sdbe_vdif.SWARM_TRANSPOSE_SIZE*(BENG_BUFFER_IN_COUNTS-1)*read_sdbe_vdif.R2DBE_RATE/read_sdbe_vdif.SWARM_RATE))
  vdif_frames_offset = 0
  rel_path_to_in = '/home/shared/sdbe_preprocessed/'
  d = sdbe_preprocess.get_diagnostics_from_file(scan_filename_base,rel_path=rel_path_to_in)
  xr = read_r2dbe_vdif.read_from_file(rel_path_to_in + scan_filename_base + '_r2dbe_eth3.vdif',N_r_vdif_frames,vdif_frames_offset)

  s_range = arange(-16,16)
  s_avg = 128
  offset_swarmdbe_data = d['offset_swarmdbe_data']
  idx_offset = d['get_idx_offset'][0]
  fft_window_size = 32768

  # compare to pre-shifted data
  hf5 = h5py.File('prep6_test1_local_sdbe_preprocess.hdf5')
  spectra = hf5.get('Xs1').value
  #spectra = np.roll(cpu_beng_spectra_1,-d['get_idx_offset'][0],axis=0)
  xs_shifted = sdbe_preprocess.resample_sdbe_to_r2dbe_fft_interp(spectra,interp_kind="linear")

  # now we check the band limited series
  xr_bl = sdbe_preprocess.bandlimit_1248_to_1024(xr[:(xr.size//4096)*4096],sub_sample=True)
  xs_bl = sdbe_preprocess.bandlimit_1248_to_1024(xs_shifted[:(xs_shifted.size//4096)*4096],sub_sample=True)


  # check correlation given pre-determined time shift

  x0 = xr_bl.copy()
  x1 = cpu_r2dbe[np.floor(idx_offset*2*BENG_CHANNELS_/SWARM_RATE*2048e6)+offset_swarmdbe_data/2-1:]
  n = np.min((2**int(np.floor(np.log2(x0.size))),2**int(np.floor(np.log2(x1.size)))))
  x0 = x0[:n]
  x1 = x1[:n]

  X0 = rfft(x0.reshape(n/fft_window_size,fft_window_size),axis=-1)
  X1 = rfft(x1.reshape(n/fft_window_size,fft_window_size),axis=-1)
  X0*= np.exp(-1.0j*np.angle(X0[0,0]))
  X1*= np.exp(-1.0j*np.angle(X1[0,0]))

  S_0x0 = (X0 * X0.conj()).mean(axis=0)
  S_1x1 = (X1 * X1.conj()).mean(axis=0)
  S_0x1 = (X0 * X1.conj()).mean(axis=0)
  p = 8; q = 16
  s_0x0 = irfft(S_0x0,n=p*fft_window_size).real
  s_1x1 = irfft(S_1x1,n=p*fft_window_size).real
  s_0x1 = irfft(S_0x1,n=p*fft_window_size).real/np.sqrt(s_0x0.max()*s_1x1.max())
  print s_0x1.max(), s_0x1.min()
  plt.figure()
  plt.stem(np.arange(-p*q/2,p*q/2),np.roll(s_0x1,p*q/2)[:p*q])

  #################

  x0 = xr_bl.copy()
  x1 = xs_bl[offset_swarmdbe_data/2-1:]
  n = np.min((2**int(np.floor(np.log2(x0.size))),2**int(np.floor(np.log2(x1.size)))))
  x0 = x0[:n]
  x1 = x1[:n]

  X0 = rfft(x0.reshape(n/fft_window_size,fft_window_size),axis=-1)
  X1 = rfft(x1.reshape(n/fft_window_size,fft_window_size),axis=-1)
  X0*= np.exp(-1.0j*np.angle(X0[0,0]))
  X1*= np.exp(-1.0j*np.angle(X1[0,0]))

  S_0x0 = (X0 * X0.conj()).mean(axis=0)
  S_1x1 = (X1 * X1.conj()).mean(axis=0)
  S_0x1 = (X0 * X1.conj()).mean(axis=0)
  p = 8; q = 16
  s_0x0 = irfft(S_0x0,n=p*fft_window_size).real
  s_1x1 = irfft(S_1x1,n=p*fft_window_size).real
  s_0x1 = irfft(S_0x1,n=p*fft_window_size).real/np.sqrt(s_0x0.max()*s_1x1.max())
  print s_0x1.max(), s_0x1.min()
  plt.figure()
  plt.stem(np.arange(-p*q/2,p*q/2),np.roll(s_0x1,p*q/2)[:p*q])

  plt.ion()
  plt.show()

print 'done!'

#0.320292665479 -0.0944922613483
#0.326504978799 -0.0429685921659

