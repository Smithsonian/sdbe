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
	// This should use shared memory....?

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
"""

kernel_module = SourceModule(kernel_source)

# generate fake B-frames
#num_vdif_frames = 39
num_vdif_frames = 64
data_shape = (num_vdif_frames*16384,128)
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

tic = get_process_cpu_time()
shift_snapshots(gpu_beng_data_1,gpu_beng_data_2,\
		block=(128,4,1),grid=(16384/4*num_vdif_frames,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

shift_snapshots_result = np.zeros_like(cpu_beng_data)
cuda.memcpy_dtoh(shift_snapshots_result,gpu_beng_data_2)

# chatter
diff0 = np.roll(cpu_beng_data,-2,axis=-1) - shift_snapshots_result
diff1 = cpu_beng_data - shift_snapshots_result
print '\nshift_snapshots test results:'
print 'shifted:',np.allclose([la.norm(diff0[i::8,:]) for i in range(4)], [0,0,0,0])
print 'same:',np.allclose([la.norm(diff1[4+i::8,:]) for i in range(4)], [0,0,0,0])
print 'shift_snapshots time:',time_gpu.nanoseconds*1e-6,' ms'

#####################################################

# reload fake B-frames
cuda.memcpy_htod(gpu_beng_data_1,cpu_beng_data)

# test shift_snapshots_inplace
shift_snapshots_inplace = kernel_module.get_function('shift_snapshots_inplace')

tic = get_process_cpu_time()
shift_snapshots_inplace(gpu_beng_data_1,\
		block=(128,4,1),grid=(16384/8*num_vdif_frames,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

cuda.memcpy_dtoh(shift_snapshots_result,gpu_beng_data_1)

# chatter
diff0 = np.roll(cpu_beng_data,-2,axis=-1) - shift_snapshots_result
diff1 = cpu_beng_data - shift_snapshots_result
print '\nshift_snapshots_inplace test results:'
print 'shifted:',np.allclose([la.norm(diff0[i::8,:]) for i in range(4)], [0,0,0,0])
print 'same:',np.allclose([la.norm(diff1[4+i::8,:]) for i in range(4)], [0,0,0,0])
print 'shift_snapshots_inplace time:',time_gpu.nanoseconds*1e-6,' ms'

#####################################################

# test shift_beng
shift_beng = kernel_module.get_function('shift_beng')

tic = get_process_cpu_time()
shift_beng(gpu_beng_data_1,gpu_beng_data_2,\
		block=(128,8,1),grid=((num_vdif_frames-1)*16384/8,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

shift_beng_result = np.zeros_like(cpu_beng_data)
cuda.memcpy_dtoh(shift_beng_result,gpu_beng_data_2)

# chatter
print '\nshift_beng test results:'
print 'shifted:',np.allclose(shift_snapshots_result[16384:,69:],shift_beng_result[:-16384,69:])
print 'same:',np.allclose(shift_snapshots_result[:-16384,:69],shift_beng_result[:-16384,:69])
print 'shift_beng time:',time_gpu.nanoseconds*1e-6,' ms'

#####################################################

# test shift_beng_transpose
shift_transpose_beng = kernel_module.get_function('shift_transpose_beng')

tic = get_process_cpu_time()
shift_transpose_beng(gpu_beng_data_1,gpu_beng_data_2,\
		block=(16,16,1),grid=((16384/16) * 128*(num_vdif_frames-1)/16,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

shift_transpose_beng_result = np.zeros((num_vdif_frames*128,16384),dtype=np.complex64)
cuda.memcpy_dtoh(shift_transpose_beng_result,gpu_beng_data_2)

print '\nshift_beng_tranpose test results:'
print np.sum([np.allclose(shift_transpose_beng_result[128*i:128*(i+1),:].T,shift_beng_result[16384*i:16384*(i+1),:]) for i in range(num_vdif_frames-1)]) == num_vdif_frames-1
print 'shift_beng_transpose time:',time_gpu.nanoseconds*1e-6,' ms'

print '\ncovering ', 1.680 * num_vdif_frames, ' ms'
