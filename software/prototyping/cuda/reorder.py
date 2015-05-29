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

__global__ void shift_snapshots(cufftComplex *beng_data,int32_t num_vdif_frames)
{
	// Shift by two snapshots every other chunk of 4 spectral channels.
	// Assumes same spectral channel for consecutive snapshots are adjacent in memory.
	// Let one block handle 1 chunk of 4 channels: blockDim.x = 128, blockDim.y = 4
	// Max number of frames covered by the grid is 2147483647 * 8 / 16384 = 1048575

	int row_id = 8*blockIdx.x + threadIdx.y;
	cufftComplex sample = beng_data[row_id * 128 + threadIdx.x];
	__syncthreads();
	beng_data[row_id*128 + ((threadIdx.x - 2) & 0x7f)] = sample;
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
	// frame id (/ 16384 * 128)
	int fid = gid >> 0x15;

        if (threadIdx.x < 69){
	  beng_data_out[gid] = beng_data_in[gid];
	} else {
	  beng_data_out[fid * 128 * 16384 + cid * 128 + threadIdx.x] =
	  	beng_data_in[(fid+1) * 128 * 16384 + cid * 128 + threadIdx.x];
	}
}
"""

kernel_module = SourceModule(kernel_source)

# generate fake data
num_vdif_frames = 39
data_shape = (num_vdif_frames*16384,128)
cpu_beng_data = (np.random.standard_normal(data_shape) + \
		1j*np.random.standard_normal(data_shape)).astype(np.complex64)

gpu_beng_data = cuda.mem_alloc(cpu_beng_data.nbytes)
cuda.memcpy_htod(gpu_beng_data,cpu_beng_data)

# test shift_snapshots
shift_snapshots = kernel_module.get_function('shift_snapshots')

tic = get_process_cpu_time()
shift_snapshots(gpu_beng_data,\
		block=(128,4,1),grid=(16384/8*num_vdif_frames,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

shift_snapshots_result = np.zeros_like(cpu_beng_data)
cuda.memcpy_dtoh(shift_snapshots_result,gpu_beng_data)

# chatter
diff0 = np.roll(cpu_beng_data,-2,axis=-1) - shift_snapshots_result
diff1 = cpu_beng_data - shift_snapshots_result
print 'shift_snapshots test results:'
print np.allclose([la.norm(diff0[i::8,:]) for i in range(4)], [0,0,0,0])
print np.allclose([la.norm(diff1[4+i::8,:]) for i in range(4)], [0,0,0,0])
print 'shift_snapshots time:',time_gpu.nanoseconds*1e-6,' ms'

# test shift_beng
shift_beng = kernel_module.get_function('shift_beng')
gpu_beng_data_final = cuda.mem_alloc(cpu_beng_data.nbytes)

tic = get_process_cpu_time()
shift_beng(gpu_beng_data,gpu_beng_data_final,\
		block=(128,8,1),grid=((num_vdif_frames-1)*16384/8,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

shift_beng_result = np.zeros_like(cpu_beng_data)
cuda.memcpy_dtoh(shift_beng_result,gpu_beng_data_final)

print 'shift_beng time:',time_gpu.nanoseconds*1e-6,' ms'
print 'shift_beng test results:'
print np.allclose(shift_snapshots_result[:-16384,:69],shift_beng_result[:-16384,:69])
print np.allclose(shift_snapshots_result[16384:,69:],shift_beng_result[:-16384,69:])
