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
data_shape = (num_beng_frames,16384,128)
cpu_beng_data = np.arange(np.prod(data_shape))
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

#reorder time: 0.64563202858  ms

#####################################################

nchan = 16384
gpu_beng_data_3 = cuda.mem_alloc(8*num_beng_frames*nchan*128)

reorderT = kernel_module.get_function('reorderT')
tic.record()
reorderT(gpu_beng_data_1,gpu_beng_data_3,np.int32(num_beng_frames),
	block=(16,16,1),grid=(128/16,16384/16,1),)
#	block=(32,32,1),grid=(128/32,16384/32,1),)
toc.record()
toc.synchronize()
time_gpu = tic.time_till(toc)
print 'reorderT time:',time_gpu,' ms'

resultT = np.zeros((num_beng_frames*128,nchan),dtype=np.complex64)
cuda.memcpy_dtoh(resultT,gpu_beng_data_3)
resultT = resultT.reshape((num_beng_frames,128,nchan))[:-1,:,:]

#reorderT time: 1.34713602066  ms

#####################################################

# compute on CPU
cpu_result = np.empty_like(cpu_beng_data)

# shift by two snapshots 
for i in range(4):
  cpu_result[:,i::8,:] = np.roll(cpu_beng_data[:,i::8,:],-2,axis=-1)
  cpu_result[:,i+4::8,:] = cpu_beng_data[:,i+4::8,:]

# then by one B-engine
cpu_result[:,:,69:] = np.roll(cpu_result[:,:,69:],-1,axis=0)
cpu_result = cpu_result[:-1,:,:]

cpu_resultT = np.empty_like(resultT)
for i in range(num_beng_frames-1):
  cpu_resultT[i,:,:] = result[i,:,:].T

print 'reorder test result:', 'pass' if np.allclose(cpu_result,result) else 'fail'
print 'reorderT test result:', 'pass' if np.allclose(cpu_resultT,resultT) else 'fail'


