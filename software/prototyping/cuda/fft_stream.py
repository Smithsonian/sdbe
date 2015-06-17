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

from timeit import default_timer as timer
from argparse import ArgumentParser
import logging
from collections import defaultdict

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

# logger
parser = ArgumentParser(description="Convert SWARM vdif data into 4096 Msps time-domain data")
parser.add_argument('-v', dest='verbose', action='store_true', help='display debug information')
parser.add_argument('-c', dest='correlate', action='store_true',help='correlate against R2DBE')
parser.add_argument('-b', dest='basename', help='scan filename base for SWARM data', default='prep6_test1_local')
parser.add_argument('-d', dest='dir', help='directory for input SWARM data products', default='/home/shared/sdbe_preprocessed/')
args = parser.parse_args()
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

# timers
clock = defaultdict(float)
counts = defaultdict(int)
total = defaultdict(float)
def tick(name):
  #clock[name] = timer()
  clock[name] = cuda.Event()
  clock[name].record()
def tock(name):
  counts[name] = counts[name] + 1
  tmp = cuda.Event()
  tmp.record()
  tmp.synchronize()
  #total[name] = total[name] + timer() - clock[name]
  total[name] = total[name] + 1e-3*tmp.time_since(clock[name])

# settings
beng_frame_offset = 1
scan_filename_base = args.basename
filename_input = args.dir+scan_filename_base+'_swarmdbe'

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

# read vdif
cpu_vdif_buf,bcount_offset = read_vdif(filename_input,num_vdif_frames,beng_frame_offset,batched=True)

# inverse FFT plan
n = array([2*BENG_CHANNELS_],int32)
inembed = array([BENG_CHANNELS],int32)
onembed = array([2*BENG_CHANNELS_],int32)
plan_A = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, BENG_CHANNELS,
                                      onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
 				       cufft.CUFFT_C2R, (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS)

batch_trim = 32
# Turn concatenated SWARM time series into single spectrum
n = array([39*2*BENG_CHANNELS_],int32)
inembed = array([39*2*BENG_CHANNELS_],int32)
onembed = array([39*BENG_CHANNELS_+1],int32)
plan_interp_A = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,39*2*BENG_CHANNELS_,
					onembed.ctypes.data,1,39*BENG_CHANNELS_+1,
					cufft.CUFFT_R2C,batch_trim)

# Turn trimmed spectrum into 2048 timeseries
n = array([32*2*BENG_CHANNELS_],int32)
inembed = array([39*BENG_CHANNELS_+1],int32)
onembed = array([32*2*BENG_CHANNELS_],int32)
plan_interp_B = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,39*BENG_CHANNELS_+1,
					onembed.ctypes.data,1,32*2*BENG_CHANNELS_,
					cufft.CUFFT_C2R,batch_trim)

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

# move cpu_vdif_buf to device
tic.record()
cuda.memcpy_htod(gpu_vdif_buf,cpu_vdif_buf)
toc.record()
toc.synchronize()
logger.info('htod transfer rate: %g GB/s' % (cpu_vdif_buf.nbytes/(tic.time_till(toc)*1e-3)))

# depacketize and dequantize BENG_BUFFER_IN_COUNTS
# gpu_beng_data_0 is USB
tic.record()
tick('depacketize')
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
tock('depacketize')

# reorder and zero pad SWARM snapshots
tick('reorder   ')
gpu_beng_0 = cuda.mem_alloc(8*BENG_CHANNELS*BENG_SNAPSHOTS*(BENG_BUFFER_IN_COUNTS-1))
gpu_beng_1 = cuda.mem_alloc(8*BENG_CHANNELS*BENG_SNAPSHOTS*(BENG_BUFFER_IN_COUNTS-1))
reorderTz_smem_kernel(gpu_beng_data_1,gpu_beng_1,np.int32(BENG_BUFFER_IN_COUNTS),
	block=(16,16,1),grid=(BENG_CHANNELS_*BENG_SNAPSHOTS/(16*16),1,1),)
reorderTz_smem_kernel(gpu_beng_data_0,gpu_beng_0,np.int32(BENG_BUFFER_IN_COUNTS),
	block=(16,16,1),grid=(BENG_CHANNELS_*BENG_SNAPSHOTS/(16*16),1,1),)
# free unpacked, unordered b-frames
gpu_beng_data_0.free()
gpu_beng_data_1.free()
tock('reorder   ')

# allocate memory for time series
tick('resample')
#gpu_r2dbe = cuda.mem_alloc(4 * num_r2dbe_samples / 2)
gpu_r2dbe = cuda.mem_alloc(4 * num_r2dbe_samples)
gpu_swarm = cuda.mem_alloc(4 * num_swarm_samples)

i_phase_sum = 0
for SB in (gpu_beng_0,gpu_beng_1):
  tick('plan_A   ')
  cufft.cufftExecC2R(plan_A,int(SB),int(gpu_swarm))
  SB.free()
  tock('plan_A   ')

  # look over chunks of 39 SWARM snapshots
  gpu_tmp = cuda.mem_alloc(8*int(39*BENG_CHANNELS_+1)*batch_trim)
  for ib in range((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS/39/batch_trim):
    # Turn concatenated SWARM time series into single spectrum
    tick('plan_interp_A')
    cufft.cufftExecR2C(plan_interp_A,int(gpu_swarm)+int(4*39*2*BENG_CHANNELS_*batch_trim*ib),int(gpu_tmp))
    tock('plan_interp_A')

    # Turn padded SWARM spectrum into time series with R2DBE sampling rate
    tick('plan_interp_B')
    cufft.cufftExecC2R(plan_interp_B,
			int(gpu_tmp)+int(8*150*512),int(gpu_r2dbe)+int(4*32*2*BENG_CHANNELS_*ib*batch_trim)+int(4*i_phase_sum*num_r2dbe_samples/2))
    tock('plan_interp_B')
  gpu_tmp.free()

  i_phase_sum += 1

tock('resample')

gpu_swarm.free()
toc.record()
toc.synchronize()
time_gpu = tic.time_till(toc)

cpu_r2dbe = np.empty((2,num_r2dbe_samples/2),dtype=float32) # empty array for result
cuda.memcpy_dtoh(cpu_r2dbe,gpu_r2dbe)
gpu_r2dbe.free()

# destroy plans
cufft.cufftDestroy(plan_A)
cufft.cufftDestroy(plan_interp_A)
cufft.cufftDestroy(plan_interp_B)

logger.info('INFO:: Time resampled: %.2f ms' % (13.128e-3 * BENG_SNAPSHOTS * (BENG_BUFFER_IN_COUNTS - 1)))
logger.info('INFO:: GPU time: %.2f ms' % time_gpu)

# report timing info
real_time = (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*2*BENG_CHANNELS_/SWARM_RATE
logger.info( "\n%s\t%s\t%s\t%s" % ('operation', 'counts', 'total [ms]', 'x(real = %.3f)' % (real_time*1e3)))
logger.info("-----------------------------------------------------------")
for k in total.keys():
  logger.info("%s\t%d\t%.2f\t\t%.3f" % (k, counts[k], 1e3*total[k], real_time/total[k]  ))

#####################################################################################

if args.correlate:
  import matplotlib.pyplot as plt
  logger.info("INFO:: Correlating result again R2DBE")

  # Now read R2DBE data covering roughly the same time window as the SWARM
  # data. Start at an offset of zero (i.e. from the first VDIF packet) to
  # keep things simple.
  N_r_vdif_frames = int(np.ceil(read_sdbe_vdif.SWARM_TRANSPOSE_SIZE*(BENG_BUFFER_IN_COUNTS-1)*read_sdbe_vdif.R2DBE_RATE/read_sdbe_vdif.SWARM_RATE))
  vdif_frames_offset = 0
  rel_path_to_in = args.dir
  d = sdbe_preprocess.get_diagnostics_from_file(scan_filename_base,rel_path=rel_path_to_in)
  xr = read_r2dbe_vdif.read_from_file(rel_path_to_in + scan_filename_base + '_r2dbe_eth3.vdif',N_r_vdif_frames,vdif_frames_offset)


  # compare to pre-shifted data
  hf5 = h5py.File(rel_path_to_in + scan_filename_base + '_sdbe_preprocess.hdf5')
  spectra = hf5.get('Xs1').value
  xs_shifted = sdbe_preprocess.resample_sdbe_to_r2dbe_fft_interp(spectra,interp_kind="linear")

  # now we check the band limited series
  xr_bl = sdbe_preprocess.bandlimit_1248_to_1024(xr[:(xr.size//4096)*4096],sub_sample=True)
  xs_bl = sdbe_preprocess.bandlimit_1248_to_1024(xs_shifted[:(xs_shifted.size//4096)*4096],sub_sample=True)

  # check correlation given pre-determined time shift
  s_range = arange(-16,16)
  s_avg = 128
  offset_swarmdbe_data = d['offset_swarmdbe_data']
  idx_offset = d['get_idx_offset'][0]
  fft_window_size = 32768

  # FFT resampled
  x0 = xr_bl.copy()
  x1 = cpu_r2dbe[1,np.floor(idx_offset*2*BENG_CHANNELS_/SWARM_RATE*2048e6)+offset_swarmdbe_data/2-1:]
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

  print 'FFT:\t\t', s_0x1.max(), s_0x1.min()
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
  print 'linear+shift:\t\t', s_0x1.max(), s_0x1.min()

  plt.figure()
  plt.stem(np.arange(-p*q/2,p*q/2),np.roll(s_0x1,p*q/2)[:p*q])

  plt.show()
