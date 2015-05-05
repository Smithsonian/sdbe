"""
Quantization kernel
"""

import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from timing import get_process_cpu_time

kernel_template = """
// Warp reduction from:
// http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

// warp reduce
__inline__ __device__ unsigned int halfWarpReduceSum(unsigned int val){
	for (int offset = warpSize/4; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

__global__ void quantize_2bit(const float *in, unsigned int *out, int N)
{
	// This kernel must be called with integer number of warps/block

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N){
	// grab sample
	int samp = (int) in[tid];

	// reinterpret sample as offset-binary
	samp  = (samp + %(OFFSET_K)d) & %(SAMP_MAX)d;

	// bit-shift using bit-masked thread id
	samp = samp << (%(BITS_PER_SAMPLE)d * (tid & 0xf)) ;

	// sum over half warps
	samp = halfWarpReduceSum(samp);

	if (tid %% 0x10 == 0) {
		out[tid / 16] = samp;
	}

	}

}
"""

def quantize(x,bits_per_sample):
	ans = np.empty(x.size * bits_per_sample / 32, dtype=np.uint32)
        samp_max = 2**bits_per_sample - 1
        samp_per_word = 32 / bits_per_sample
        for word_n in range(ans.size):
            word = 0

            for samp_n in range(samp_per_word):
                samp = int(x[word_n * samp_per_word + samp_n])

                # reinterpret sample as offset-binary
                samp = (samp + 2**(bits_per_sample-1)) & samp_max

                # add the sample data to the word
                shift_by = bits_per_sample * samp_n
                word = word + (samp << shift_by)

            ans[word_n] = word

        return ans

# settings
NUM_SAMPLES = 2**20
BITS_PER_SAMPLE = 2

kernel_source = kernel_template % {'BITS_PER_SAMPLE':BITS_PER_SAMPLE,'OFFSET_K':2**(BITS_PER_SAMPLE-1),'SAMP_MAX':2**BITS_PER_SAMPLE-1}

kernel_module = SourceModule(kernel_source)

# create floating point array
h_32bit_signal = np.random.standard_normal(NUM_SAMPLES).astype(np.float32)

#
d_32bit_signal = cuda.mem_alloc(h_32bit_signal.nbytes)
cuda.memcpy_htod(d_32bit_signal,h_32bit_signal)

#
quantize_2bit = kernel_module.get_function('quantize_2bit')

#
d_2bit_signal = cuda.mem_alloc(h_32bit_signal.size * 2 / 8)

# quantize
tic = get_process_cpu_time()
quantize_2bit(d_32bit_signal,d_2bit_signal,np.int32(NUM_SAMPLES),\
	block=(512,1,1),grid=(NUM_SAMPLES/512,1))
#	block=(32,1,1),grid=(NUM_SAMPLES/32,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

# pull back answer
h_2bit_signal = np.empty(NUM_SAMPLES / 16, dtype=np.uint32)
cuda.memcpy_dtoh(h_2bit_signal,d_2bit_signal)

# compute on CPU
tic = get_process_cpu_time()
_2bit_signal = quantize(h_32bit_signal,2)
toc = get_process_cpu_time()
time_cpu = toc-tic

print 'gpu time:',time_gpu
print 'cpu time:',time_cpu
if np.allclose(_2bit_signal, h_2bit_signal):
	print 'test passed'
else:
	print 'test passed'
