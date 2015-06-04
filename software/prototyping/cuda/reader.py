'''
Assumes reader.cu compiled with BENG_FRAMES_OUT_CONSECUTIVE_SNAPSHOTS
'''

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from numpy import complex64,float32,int32,uint32,array,arange,empty,zeros
from struct import unpack

from timing import get_process_cpu_time
from time import time

import numpy as np

reader_template = """
/*
 * GPU kernel code for the following pipeline components:
 *   VDIF interpreter
 *   B-engine depacketizer
 *   Pre-preprocessor
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
 int32_t bcount_offset,
 int blocks_per_grid)
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

 for (iframe=0; iframe + threadIdx.y + blockIdx.x*blockDim.y<num_vdif_frames; iframe+=blocks_per_grid*blockDim.y)
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
  old = atomicAdd(beng_frame_completion + ((bcount-bcount_offset)&(4 -1)), 1);
# 1277 "reader.cu"
  if (__any(old == ((16384/8)*blockDim.x)-1))
  {
# 1298 "reader.cu"
  }
 }
}
"""

VDIF_BYTE_SIZE = 1056
VDIF_BYTE_SIZE_DATA = 1024
VDIF_INT_SIZE = VDIF_BYTE_SIZE/4
VDIF_PER_BENG = 2048

BENG_CHANNELS_ = 16384
BENG_CHANNELS = (BENG_CHANNELS_ + 1)
BENG_SNAPSHOTS = 128
#BENG_BUFFER_IN_COUNTS = 4
#BENG_BUFFER_IN_COUNTS = 64
BENG_BUFFER_IN_COUNTS = 40

def meminfo(kernel):
  print "Registers: %d" % kernel.num_regs
  print "Local: %d" % kernel.local_size_bytes
  print "Shared: %d" % kernel.shared_size_bytes
  print "Const: %d" % kernel.const_size_bytes

def get_bcount_from_vdif(vdif_start):
  BENG_VDIF_HDR_0_OFFSET_INT = 4  # b1 b2 b3 b4
  BENG_VDIF_HDR_1_OFFSET_INT = 5  # //  c  0  f b0
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

def read_datafile(filename_data):
  fh_data = open(filename_data,'r')
  # read buffer_size
  beng_buffer_in_counts = unpack('I1',fh_data.read(4))[0]
  # read B-engine completion counters
  beng_frame_completion = array(unpack('{0}I'.format(beng_buffer_in_counts),fh_data.read(beng_buffer_in_counts*4))) 
  # read B-engine data for phased sum 0
  tmp = array(unpack('{0}f'.format(2*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS),fh_data.read(8*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS)),float32)
  beng_data_0 = (tmp[::2] + 1j * tmp[1::2])
  beng_data_0.resize((BENG_BUFFER_IN_COUNTS,BENG_CHANNELS_,BENG_SNAPSHOTS))
  # skip 8*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS bytes (reader.cu reads in 16384 channels but allocates for 16485).
  fh_data.seek(8*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS,1)
  # read B-engine data for phased sum 1
  tmp = array(unpack('{0}f'.format(2*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS),fh_data.read(8*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS)),float32)
  beng_data_1 = (tmp[::2] + 1j * tmp[1::2])
  beng_data_1.resize((BENG_BUFFER_IN_COUNTS,BENG_CHANNELS_,BENG_SNAPSHOTS))
  fh_data.close()
  return {'beng_buffer_in_counts':beng_buffer_in_counts,'beng_data_0':beng_data_0,'beng_data_1':beng_data_1}


########################################################################

# settings
filename_input = '/home/shared/sdbe_preprocessed/prep6_test1_local_swarmdbe'
beng_frame_offset = 1
num_vdif_frames = BENG_BUFFER_IN_COUNTS*VDIF_PER_BENG #
repeats = 2
# ./reader.out -I /home/shared/sdbe_preprocessed/prep6_test1_local_swarmdbe -B 1 -v -c 8192 -d c8192_B1.bin 
#filename_data = 'c8192_B1.bin'
# ./reader.out -I /home/shared/sdbe_preprocessed/prep6_test1_local_swarmdbe -B 1 -v -c 16384 -d c16384_B1.bin 
#filename_data = 'c16384_B1.bin'
filename_data = ''

# read in vdif
vdif_buf,bcount_offset = read_vdif(filename_input,num_vdif_frames,beng_frame_offset,batched=True)

# interpret and depacket VDIF
print '\nvdif_to_beng:'
reader_source = reader_template % {'BENG_BUFFER_IN_COUNTS':BENG_BUFFER_IN_COUNTS}
reader_module = SourceModule(reader_source)
vdif_to_beng = reader_module.get_function('vdif_to_beng') # 34 regs
meminfo(vdif_to_beng)

# allocate memory on device 
beng_data_bytes = 8 * BENG_CHANNELS_ * BENG_SNAPSHOTS * BENG_BUFFER_IN_COUNTS
gpu_vdif_buf = cuda.mem_alloc(vdif_buf.nbytes)
gpu_beng_data_0 = cuda.mem_alloc(beng_data_bytes)
gpu_beng_data_1 = cuda.mem_alloc(beng_data_bytes)
gpu_fid = cuda.mem_alloc(4*num_vdif_frames)
gpu_cid = cuda.mem_alloc(4*num_vdif_frames)
gpu_bcount = cuda.mem_alloc(4*num_vdif_frames)
gpu_beng_frame_completion = cuda.mem_alloc(4*BENG_BUFFER_IN_COUNTS)

# copy vdif data to device
beng_frame_completion = zeros(BENG_BUFFER_IN_COUNTS,dtype=int32)
cuda.memcpy_htod(gpu_vdif_buf,vdif_buf)
cuda.memcpy_htod(gpu_beng_frame_completion,beng_frame_completion)

blocks_per_grid = 128
for ir in arange(repeats): 
  tic = get_process_cpu_time()
  tick = time()
  vdif_to_beng(gpu_vdif_buf, gpu_fid, gpu_cid, gpu_bcount, gpu_beng_data_0, gpu_beng_data_1, gpu_beng_frame_completion,int32(num_vdif_frames),int32(bcount_offset),int32(blocks_per_grid),
  	block=(32,32,1), grid=(blocks_per_grid,1,1))
  cuda.Stream(0).synchronize()
  toc = get_process_cpu_time()
  tock = time()
  time_gpu = toc - tic
  print 'CPU:',time_gpu.nanoseconds*1e-6,' ms'
  time_gpu = tock - tick
  print 'python:',time_gpu*1e3,' ms'

# retrieve depacketized B-engine
beng_data_0 = empty(beng_data_bytes/8,dtype=complex64)
beng_data_1 = empty(beng_data_bytes/8,dtype=complex64)
cuda.memcpy_dtoh(beng_data_0,gpu_beng_data_0)
cuda.memcpy_dtoh(beng_data_1,gpu_beng_data_1)
cuda.memcpy_dtoh(beng_frame_completion,gpu_beng_frame_completion)
beng_data_0.resize((BENG_BUFFER_IN_COUNTS,BENG_CHANNELS_,BENG_SNAPSHOTS))
beng_data_1.resize((BENG_BUFFER_IN_COUNTS,BENG_CHANNELS_,BENG_SNAPSHOTS))

print '\nvdif_to_beng test results:'
if filename_data != '':
  fh_data = read_datafile(filename_data)
  print np.allclose(beng_data_0,fh_data['beng_data_0'])
  print np.allclose(beng_data_1,fh_data['beng_data_1'])
else:
  print np.unique(beng_data_0).size == 16
  print np.unique(beng_data_1).size == 16

# free memory
gpu_vdif_buf.free()
gpu_fid.free()
gpu_cid.free()
gpu_bcount.free()
gpu_beng_data_0.free()
gpu_beng_data_1.free()
gpu_beng_frame_completion.free()
