import numpy as np
import numpy.linalg as la

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

kernel_source = """
#include <cufft.h>

__global__ void reorder(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){

  // gridDim.y = 16384 / 16 = 1024
  // gridDim.x = 128 / 16 = 8
  // blockDim.x = 16; blockDim.y = 16;
  // --> launches 2097152 threads

  //int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  //int threadId = blockId*blockDim.x + threadIdx.x;
  int sid_out,fid_in;

  // snapshot id 
  int sid_in = threadIdx.x + blockDim.x*blockIdx.x;
  // channel id
  int cid = threadIdx.y + blockDim.y*blockIdx.y;

  // shift by 2-snapshots case:
  if (((cid / 4) & (0x1)) == 0) {
    sid_out = (sid_in-2) & 0x7f;
  } else {
    sid_out = sid_in;
  }

  // for now, let us loop the grid over B-engine frames:
  for (int fid_out=0; fid_out<num_beng_frames-1; fid_out+=1){
    if (sid_out < 69){
      fid_in = fid_out;
    } else {
      fid_in = fid_out+1;
    }
    beng_data_out[128*16384*fid_out + 128*cid + sid_out] = beng_data_in[128*16384*fid_in + 128*cid + sid_in];
  }
}

__global__ void reorderT(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){

  // gridDim.y = 16384 / 16 = 1024
  // gridDim.x = 128 / 16 = 8
  // blockDim.x = 16; blockDim.y = 16;
  // --> launches 2097152 threads

  //int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  //int threadId = blockId*blockDim.x + threadIdx.x;
  int sid_out,fid_in;

  // snapshot id 
  int sid_in = threadIdx.x + blockDim.x*blockIdx.x;
  // channel id
  int cid = threadIdx.y + blockDim.y*blockIdx.y;

  // shift by 2-snapshots case:
  if (((cid / 4) & (0x1)) == 0) {
    sid_out = (sid_in-2) & 0x7f;
  } else {
    sid_out = sid_in;
  }

  // for now, let us loop the grid over B-engine frames:
  for (int fid_out=0; fid_out<num_beng_frames-1; fid_out+=1){
    if (sid_out < 69){
      fid_in = fid_out;
    } else {
      fid_in = fid_out+1;
    }
    beng_data_out[128*16384*fid_out + 16384*sid_out + cid] = beng_data_in[128*16384*fid_in + 128*cid + sid_in];
  }
}
"""

# two timers for speed-testing
tic = cuda.Event()
toc = cuda.Event()

kernel_module = SourceModule(kernel_source)

# generate fake B-frames
num_beng_frames = 4
data_shape = (num_beng_frames*16384,128)
cpu_beng_data = np.arange(data_shape[0] * data_shape[1])
cpu_beng_data = cpu_beng_data.reshape(data_shape).astype(np.complex64)

gpu_beng_data_1 = cuda.mem_alloc(cpu_beng_data.nbytes)
gpu_beng_data_2 = cuda.mem_alloc(cpu_beng_data.nbytes)
cuda.memcpy_htod(gpu_beng_data_1,cpu_beng_data)

#####################################################

reorder = kernel_module.get_function('reorder')
tic.record()
reorder(gpu_beng_data_1,gpu_beng_data_2,np.int32(num_beng_frames),
	block=(16,16,1),grid=(8,1024,1),)
toc.record()
toc.synchronize()
time_gpu = tic.time_till(toc)
print 'reorder time:',time_gpu,' ms'

result = np.zeros((num_beng_frames*16384,128),dtype=np.complex64)
cuda.memcpy_dtoh(result,gpu_beng_data_2)
result = result.reshape((num_beng_frames,16384,128))[:-1,:,:]

#####################################################

reorderT = kernel_module.get_function('reorder')
tic.record()
reorderT(gpu_beng_data_1,gpu_beng_data_2,np.int32(num_beng_frames),
	block=(16,16,1),grid=(8,1024,1),)
toc.record()
toc.synchronize()
time_gpu = tic.time_till(toc)
print 'reorderT time:',time_gpu,' ms'

resultT = np.zeros((num_beng_frames*128,16384),dtype=np.complex64)
cuda.memcpy_dtoh(resultT,gpu_beng_data_2)
resultT = resultT.reshape((num_beng_frames,128,16384))[:-1,:,:]
