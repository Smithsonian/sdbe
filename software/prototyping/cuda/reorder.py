import numpy as np
import numpy.linalg as la

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from timing import get_process_cpu_time

kernel_source = """
#include <cufft.h>

__global__ void reorder(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){

  int32_t sid_out,bid_out;

  // use a  linear index
  int32_t gid = blockIdx.x*blockDim.x + threadIdx.x;
  // to calculate the snapshot id,
  int32_t sid_in = gid % 128;
  // the B-counter id,
  int32_t bid_in = (gid % (128 * num_beng_frames)) / 128;
  // and the channel id.
  int32_t cid = (gid  / (128 * num_beng_frames));

  // shift-by-2 snapshots case:
  if (((cid / 4) & (0x1)) == 0) {
    sid_out = (sid_in-2) & 0x7f;
  } else {
    sid_out = sid_in;
  }

  // and shift-by-1 B-engine counter case:
  if (sid_out < 69){
    bid_out = bid_in;
  } else {
    bid_out = (bid_in-1);
  }

  if (bid_out > -1) {
    beng_data_out[128*num_beng_frames*cid + 128*bid_out + sid_out] = beng_data_in[gid];
  }
}

__global__ void reorderT(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){
  int32_t sid_out,bid_out;
  // use a  linear index
  int32_t gid = blockIdx.x*blockDim.x + threadIdx.x;
  // to calculate the snapshot id,
  int32_t sid_in = gid % 128;
  // the B-counter id,
  int32_t bid_in = (gid % (128 * num_beng_frames)) / 128;
  // and the channel id.
  int32_t cid = (gid  / (128 * num_beng_frames));

  // shift-by-2 snapshots case:
  if (((cid / 4) & (0x1)) == 0) {
    sid_out = (sid_in-2) & 0x7f;
  } else {
    sid_out = sid_in;
  }

  // and shift-by-1 B-engine counter case:
  if (sid_out < 69){
    bid_out = bid_in;
  } else {
    bid_out = (bid_in-1);
  }

  if (bid_out > -1) {
  beng_data_out[128*16384*bid_out + 16384*sid_out + cid] = 
			beng_data_in[gid];
  }
}

__global__ void reorderT_smem(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){

  // gridDim.x = 16384 * 128 / (16 * 16) = 8192
  // blockDim.x = 16; blockDim.y = 16;
  // --> launches 2097152 threads

  int32_t sid_out,bid_in;

  __shared__ cufftComplex tile[16][16];

  // for now, let us loop the grid over B-engine frames:
  for (int bid_out=0; bid_out<num_beng_frames-1; bid_out+=1){

    // input snapshot id
    int sid_in = (blockIdx.x * blockDim.x + threadIdx.x) % 128;
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
    sid_in = threadIdx.y + (blockIdx.x*blockDim.y) % 128;
    // channel id 
    cid = threadIdx.x + blockDim.x * (blockIdx.x / (128 / blockDim.x)); 

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    beng_data_out[128*16384*bid_out + 16384*sid_out + cid] = tile[threadIdx.y][threadIdx.x];

    __syncthreads();
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
    int sid_in = (blockIdx.x * blockDim.x + threadIdx.x) % 128;
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
    sid_in = threadIdx.y + (blockIdx.x*blockDim.y) % 128;
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

# two timers for speed-testing
tic = cuda.Event()
toc = cuda.Event()

kernel_module = SourceModule(kernel_source)

# generate fake B-frames
num_beng_frames = 40
data_shape = (16384,num_beng_frames,128)
cpu_beng_data = np.arange(np.prod(data_shape))
cpu_beng_data = cpu_beng_data.reshape(data_shape).astype(np.complex64)

gpu_beng_data_1 = cuda.mem_alloc(cpu_beng_data.nbytes)
gpu_beng_data_2 = cuda.mem_alloc(cpu_beng_data.nbytes)
cuda.memcpy_htod(gpu_beng_data_1,cpu_beng_data)
repeats = 3

#####################################################

reorder = kernel_module.get_function('reorder')

for ir in range(repeats):
  tic.record()
  reorder(gpu_beng_data_1,gpu_beng_data_2,np.int32(num_beng_frames),
	block=(1024,1,1),grid=(np.prod(data_shape) / 1024,1))
  toc.record()
  toc.synchronize()
  time_gpu = tic.time_till(toc)
  print 'reorder time:',time_gpu,' ms'

result = np.zeros(np.prod(data_shape),dtype=np.complex64)
cuda.memcpy_dtoh(result,gpu_beng_data_2)
result = result.reshape(data_shape)[:,:-1,:]

#reorder time: 0.6  ms

#####################################################
reorderT = kernel_module.get_function('reorderT')

for ir in range(repeats):
  tic.record()
  reorderT(gpu_beng_data_1,gpu_beng_data_2,np.int32(num_beng_frames),
	block=(1024,1,1),grid=(np.prod(data_shape) / 1024,1))
  toc.record()
  toc.synchronize()
  time_gpu = tic.time_till(toc)
  print 'reorderT time:',time_gpu,' ms'

resultT = np.zeros((num_beng_frames,128,16384),dtype=np.complex64)
cuda.memcpy_dtoh(resultT,gpu_beng_data_2)
resultT = resultT[:-1,:,:]
#####################################################

reorderT_smem = kernel_module.get_function('reorderT_smem')
gpu_beng_data_3 = cuda.mem_alloc(8*16385*128*num_beng_frames)

for ir in range(repeats):
  tic.record()
  reorderT_smem(gpu_beng_data_1,gpu_beng_data_3,np.int32(num_beng_frames),
	block=(16,16,1),grid=(16384*128/(16*16),1))
  toc.record()
  toc.synchronize()
  time_gpu = tic.time_till(toc)
  print 'reorderT_smem time:',time_gpu,' ms'

resultT_smem = np.zeros((num_beng_frames,128,16384),dtype=np.complex64)
cuda.memcpy_dtoh(resultT_smem,gpu_beng_data_3)
resultT_smem = resultT_smem[:-1,:,:]

#####################################################
reorderTz_smem = kernel_module.get_function('reorderTz_smem')

for ir in range(repeats):
  tic.record()
  reorderTz_smem(gpu_beng_data_1,gpu_beng_data_3,np.int32(num_beng_frames),
	block=(16,16,1),grid=(16384*128/(16*16),1))
  toc.record()
  toc.synchronize()
  time_gpu = tic.time_till(toc)
  print 'reorderTz_smem time:',time_gpu,' ms'

resultTz_smem = np.zeros((num_beng_frames,128,16385),dtype=np.complex64)
cuda.memcpy_dtoh(resultTz_smem,gpu_beng_data_3)
resultTz_smem = resultTz_smem[:-1,:,:]

#####################################################
# compute on CPU
cpu_result = np.empty_like(cpu_beng_data)
cpu_resultT = np.empty_like(resultT)

tick = get_process_cpu_time()

# shift by two snapshots 
for i in range(4):
  cpu_result[i::8,:,:] = np.roll(cpu_beng_data[i::8,:,:],-2,axis=-1)
  cpu_result[i+4::8,:,:] = cpu_beng_data[i+4::8,:,:]

# then by one B-engine
cpu_result[:,:,69:] = np.roll(cpu_result[:,:,69:],-1,axis=-2)

# ignore last B-frame
cpu_result = cpu_result[:,:-1,:]

# transpose for reorderT test
for i in range(num_beng_frames-1):
  cpu_resultT[i,:,:] = result[:,i,:].T

tock = get_process_cpu_time()
time_cpu = tock - tick

print '\ncompare vs numpy calculation (%f ms):' % (time_cpu.nanoseconds*1e-6)
print 'reorder test result:', 'pass' if np.all(cpu_result==result) else 'fail'
print 'reorderT test result:', 'pass' if np.all(cpu_resultT==resultT) else 'fail'
print 'reorderT_smem test result:', 'pass' if np.all(cpu_resultT==resultT_smem) else 'fail'
print 'reorderTz_smem test result:', 'pass' if np.all(cpu_resultT==resultTz_smem[:,:,:-1]) else 'fail'
print 'reorderTz_smem test result:', 'pass' if np.all(resultTz_smem[:,:,-1] == 0) else 'fail'

print '\nprocessed time: %f ms' % (1.680*(num_beng_frames-1))
