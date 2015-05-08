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
__inline__ __device__ unsigned int partialWarpReduceSum(unsigned int val){
	//for (int offset = warpSize/4; offset > 0; offset /= 2)
	for (int offset = warpSize/(2*%(BITS_PER_SAMPLE)d); offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

__global__ void quantize(const float *in, unsigned int *out, int N)
{
	// This kernel must be called with integer number of warps/block

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N){
	// grab sample
	int samp = (int) in[tid];

	// reinterpret sample as offset-binary
	samp  = (samp + %(OFFSET_K)d) & %(SAMP_MAX)d;

	// bit-shift using bit-masked thread id
	samp = samp << (%(BITS_PER_SAMPLE)d * (tid & %(BIT_MASK)s)) ;

	// sum over partial warps
	samp = partialWarpReduceSum(samp);

	if (tid %% (32 / %(BITS_PER_SAMPLE)d) == 0) {
		out[tid / (32 / %(BITS_PER_SAMPLE)d)] = samp;
	}

	}

}
"""

def cpu_quantize(x,bits_per_sample):
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

# use kernel template to generate source code
kernel_source = kernel_template % {'BITS_PER_SAMPLE':BITS_PER_SAMPLE,'OFFSET_K':2**(BITS_PER_SAMPLE-1),'SAMP_MAX':2**BITS_PER_SAMPLE-1, 'BIT_MASK':hex(32 / BITS_PER_SAMPLE - 1)}

# compile
kernel_module = SourceModule(kernel_source)

# create floating point array
h_32bit_signal = np.random.standard_normal(NUM_SAMPLES).astype(np.float32)

# from float32 signal from cpu to gpu
d_32bit_signal = cuda.mem_alloc(h_32bit_signal.nbytes)
cuda.memcpy_htod(d_32bit_signal,h_32bit_signal)

# get kernel function
gpu_quantize = kernel_module.get_function('quantize')

# allocate device memory for quantized signal
d_q_signal = cuda.mem_alloc(h_32bit_signal.size * BITS_PER_SAMPLE / 8)

# quantize on gpu
tic = get_process_cpu_time()
gpu_quantize(d_32bit_signal,d_q_signal,np.int32(NUM_SAMPLES),\
	block=(512,1,1),grid=(NUM_SAMPLES/512,1))
#	block=(32,1,1),grid=(NUM_SAMPLES/32,1))
toc = get_process_cpu_time()
time_gpu = toc - tic

# pull back answer to cpu
h_q_signal = np.empty(NUM_SAMPLES / (32 / BITS_PER_SAMPLE), dtype=np.uint32)
cuda.memcpy_dtoh(h_q_signal,d_q_signal)

# quantize on CPU
tic = get_process_cpu_time()
_q_signal = cpu_quantize(h_32bit_signal,BITS_PER_SAMPLE)
toc = get_process_cpu_time()
time_cpu = toc-tic

# report timing and check answer
print 'gpu time:',time_gpu
print 'cpu time:',time_cpu
if np.allclose(_q_signal, h_q_signal):
	print 'test passed'
else:
	print 'test failed' 
