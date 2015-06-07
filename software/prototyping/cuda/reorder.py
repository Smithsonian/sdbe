"""
Reorder B-frames kernel
"""

import numpy as np
import numpy.linalg as la

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from timing import get_process_cpu_time

kernel_source = """
#include <cufft.h>

__global__ void shift_snapshots_inplace(cufftComplex *beng_data)
{
	// Shift in-place by two snapshots every other chunk of 4 spectral channels.
	// Assumes same spectral channel for consecutive snapshots are adjacent in memory.
	// Let one block handle 1 chunk of 4 channels: blockDim.x = 128, blockDim.y = 4
	// Max number of frames covered by the grid is 2147483647 * 8 / 16384 = 1048575

	int row_id = 8*blockIdx.x + threadIdx.y;
	cufftComplex sample = beng_data[row_id * 128 + threadIdx.x];
	__syncthreads();
	beng_data[row_id*128 + ((threadIdx.x - 2) & 0x7f)] = sample;
}

__global__ void shift_snapshots(cufftComplex *beng_data_in, cufftComplex *beng_data_out)
{
	// Shift by two snapshots every other chunk of 4 spectral channels.
	// Assumes same spectral channel for consecutive snapshots are adjacent in memory.
	// Let one block handle 1 chunk of 4 channels: blockDim.x = 128, blockDim.y = 4
	// Max number of frames covered by the grid is 2147483647 * 4 / 16384 = 524287

	int row_id = 4*blockIdx.x + threadIdx.y;
	if ((blockIdx.x & 0x1) == 0) {
	  beng_data_out[128*row_id + ((threadIdx.x - 2) & 0x7f)] = beng_data_in[128*row_id + threadIdx.x ];
	} else {
	  beng_data_out[128*row_id + threadIdx.x ] = beng_data_in[128*row_id + threadIdx.x ];
	}
}

__global__ void shift_beng(cufftComplex *beng_data_in, cufftComplex *beng_data_out)
{
	// Shift snapshots 69-127 by one B-engine frame
	// Assumes same spectral channel for consecutive snapshots are adjacent in memory.
	// blockDim.x = 128, blockDim.y <= 8

	// global thread id
	int gid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	// channel id (gid / 128 mod 16384)
	int cid = (gid >> 0x7) & 0x3fff;
	// frame id (/ (16384 * 128) )
	int fid = gid >> 0x15;

        if (threadIdx.x < 69){
	  beng_data_out[gid] = beng_data_in[gid];
	} else {
	  beng_data_out[fid * 128 * 16384 + cid * 128 + threadIdx.x] =
	  	beng_data_in[(fid+1) * 128 * 16384 + cid * 128 + threadIdx.x];
	}
}

__global__ void shift_transpose_beng(cufftComplex *beng_data_in, cufftComplex *beng_data_out){
	// Shift snapshots 69-127 by one B-engine frame
	// Assumes same spectral channel for consecutive snapshots are adjacent in memory.
	// Outputs with consectutive channels for same snapshot adjacent in memory (conducive for batched FT).
	// blockDim.x = 16, blockDim.y = 16
	// This should use shared memory.

	// global thread id
	int gid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	// snapshot id ( blockDim.x * (blockIdx.x mod (128 / blockDim.x)) + threadIdx.x )
	int sid = blockDim.x * (blockIdx.x & (128 / blockDim.x - 1)) + threadIdx.x;
	// channel id ( threadIdx.y + blockDim.y * (blockIdx.x / (128 / blockDim.x)) mod (16384 / blockDim.y)) )
	int cid = threadIdx.y + blockDim.y * ((blockIdx.x / (128 / blockDim.x)) & (16384 / blockDim.y - 1));
	// frame id (gid / (16384 * 128) )
	int fid = gid >> 0x15;

        if (sid < 69){
	  beng_data_out[128 * 16384 * fid + 16384 * sid + cid] = beng_data_in[ fid * 128 * 16384 + cid * 128 + sid];
	} else {
	  beng_data_out[128 * 16384 * fid + 16384 * sid + cid] = beng_data_in[ (fid+1) * 128 * 16384 + cid * 128 + sid];
	}
}

__global__ void zero_pad(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int32_t num_vdif_frames, int32_t n){
  int32_t idata, iframe;
  // thread y index loops over snapshots.
  for (iframe=threadIdx.y; iframe<num_vdif_frames; iframe+=blockDim.y){	  
    // thread x index loops over output channels.
    for (idata=threadIdx.x+blockIdx.x*blockDim.x; idata<n; idata+=gridDim.x*blockDim.x){
      if (idata < 16384){
        beng_data_out[iframe*n+idata] = beng_data_in[iframe*16384 + idata];
      }
      else {
        beng_data_out[iframe*n+idata] = make_cuComplex(0.,0.);
      }
    }
  }
}
"""

# two timers for speed-testing
tic = cuda.Event()
toc = cuda.Event()

kernel_module = SourceModule(kernel_source)

# generate fake B-frames
num_beng_frames = 39
data_shape = (num_beng_frames*16384,128)
#cpu_beng_data = (np.random.standard_normal(data_shape) + \
#		1j*np.random.standard_normal(data_shape)).astype(np.complex64)
cpu_beng_data = np.arange(data_shape[0] * data_shape[1])
cpu_beng_data = cpu_beng_data.reshape(data_shape).astype(np.complex64)
	

gpu_beng_data_1 = cuda.mem_alloc(cpu_beng_data.nbytes)
cuda.memcpy_htod(gpu_beng_data_1,cpu_beng_data)

#####################################################

# test shift_snapshots
shift_snapshots = kernel_module.get_function('shift_snapshots')
gpu_beng_data_2 = cuda.mem_alloc(cpu_beng_data.nbytes)

tick = get_process_cpu_time()
tic.record()
shift_snapshots(gpu_beng_data_1,gpu_beng_data_2,\
		block=(128,4,1),grid=(16384/4*num_beng_frames,1))
#cuda.Stream().synchronize()
toc.record()
toc.synchronize()
tock = get_process_cpu_time()
time_gpu = tic.time_till(toc)
time_cpu = tock - tick

shift_snapshots_result = np.zeros_like(cpu_beng_data)
cuda.memcpy_dtoh(shift_snapshots_result,gpu_beng_data_2)

# chatter
diff0 = np.roll(cpu_beng_data,-2,axis=-1) - shift_snapshots_result
diff1 = cpu_beng_data - shift_snapshots_result
print '\nshift_snapshots test results:'
print 'shifted:',np.allclose([la.norm(diff0[i::8,:]) for i in range(4)], [0,0,0,0])
print 'same:',np.allclose([la.norm(diff1[4+i::8,:]) for i in range(4)], [0,0,0,0])
print 'CPU:',time_cpu.nanoseconds*1e-6,' ms'
print 'GPU:',time_gpu,' ms'

#####################################################

# reload fake B-frames
cuda.memcpy_htod(gpu_beng_data_1,cpu_beng_data)

#####################################################
print '\nshift_snapshots_inplace test results:'
# test shift_snapshots_inplace
shift_snapshots_inplace = kernel_module.get_function('shift_snapshots_inplace')

tick = get_process_cpu_time()
tic.record()
shift_snapshots_inplace(gpu_beng_data_1,\
		block=(128,4,1),grid=(16384/8*num_beng_frames,1))
toc.record()
toc.synchronize()
tock = get_process_cpu_time()
time_cpu = tock - tick
time_gpu = tic.time_till(toc)

cuda.memcpy_dtoh(shift_snapshots_result,gpu_beng_data_1)

# chatter
diff0 = np.roll(cpu_beng_data,-2,axis=-1) - shift_snapshots_result
diff1 = cpu_beng_data - shift_snapshots_result
print 'shifted:',np.allclose([la.norm(diff0[i::8,:]) for i in range(4)], [0,0,0,0])
print 'same:',np.allclose([la.norm(diff1[4+i::8,:]) for i in range(4)], [0,0,0,0])
print 'CPU:',time_cpu.nanoseconds*1e-6,' ms'
print 'GPU:',time_gpu,' ms'

#####################################################

# test shift_beng
print '\nshift_beng test results:'
shift_beng = kernel_module.get_function('shift_beng')

tick = get_process_cpu_time()
tic.record()
shift_beng(gpu_beng_data_1,gpu_beng_data_2,\
		block=(128,8,1),grid=((num_beng_frames-1)*16384/8,1))
toc.record()
toc.synchronize()
tock = get_process_cpu_time()
time_cpu = tock - tick
time_gpu = tic.time_till(toc)

shift_beng_result = np.zeros_like(cpu_beng_data)
cuda.memcpy_dtoh(shift_beng_result,gpu_beng_data_2)

print 'shifted:',np.allclose(shift_snapshots_result[16384:,69:],shift_beng_result[:-16384,69:])
print 'same:',np.allclose(shift_snapshots_result[:-16384,:69],shift_beng_result[:-16384,:69])
print 'CPU:',time_cpu.nanoseconds*1e-6,' ms'
print 'GPU:',time_gpu,' ms'

#####################################################

# test shift_beng_transpose
print '\nshift_beng_tranpose test results:'
shift_transpose_beng = kernel_module.get_function('shift_transpose_beng')

tick = get_process_cpu_time()
tic.record()
shift_transpose_beng(gpu_beng_data_1,gpu_beng_data_2,\
		block=(16,16,1),grid=((16384/16)*128*(num_beng_frames-1)/16,1))
toc.record()
toc.synchronize()
tock = get_process_cpu_time()
time_cpu = tock - tick
time_gpu = tic.time_till(toc)

shift_transpose_beng_result = np.zeros((num_beng_frames*128,16384),dtype=np.complex64)
cuda.memcpy_dtoh(shift_transpose_beng_result,gpu_beng_data_2)

print np.sum([np.allclose(shift_transpose_beng_result[128*i:128*(i+1),:].T,shift_beng_result[16384*i:16384*(i+1),:]) for i in range(num_beng_frames-1)]) == num_beng_frames-1
print 'CPU:',time_cpu.nanoseconds*1e-6,' ms'
print 'GPU:',time_gpu,' ms'

#####################################################
# test shift_beng_transpose
print '\nzero_pad results'

zero_pad = kernel_module.get_function('zero_pad')
blocks_per_grid = 128
num_vdif_frames = np.int32(num_beng_frames*128)

for ir in range(5):

  for n in np.array([16400,16416,16385]):

    gpu_beng_data_3 = cuda.mem_alloc(8*num_vdif_frames*n)

    tic.record()
    zero_pad(gpu_beng_data_2,gpu_beng_data_3,num_vdif_frames,np.int32(n),
		block=(32,32,1),grid=(blocks_per_grid,1))
    toc.record()
    toc.synchronize()
    time_gpu = tic.time_till(toc)
    print 'GPU:',time_gpu,' ms'
    gpu_beng_data_3.free()

  #zero_pad_result = np.empty((num_vdif_frames,n),dtype=np.complex64)
  #cuda.memcpy_dtoh(zero_pad_result,gpu_beng_data_3)
  #print np.sum(zero_pad_result != np.hstack([ shift_transpose_beng_result, np.zeros((num_vdif_frames, n - 16384), dtype=np.complex64)])) == 0

print '\ncovering ', 1.680 * num_beng_frames, ' ms'
