"""
Define utilities for handling SWARM DBE data product using
GPU acceleration
"""

from numpy import empty, zeros, int8, array, concatenate, floor, arange, \
	roll, complex64, float32, hstack, ceil, int32, float64
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import scikits.cuda.fft as cu_fft
from datetime import datetime
from timing import get_process_cpu_time, CLOCK_RES_PROCESS_CPU
from numpy.random import standard_normal

# VDIF frame size
FRAME_SIZE_BYTES = 1056

# SWARM related constants, should probably be imported from some python
# source in the SWARM git repo
SWARM_XENG_PARALLEL_CHAN = 8
SWARM_N_INPUTS = 2
SWARM_N_FIDS = 8
SWARM_TRANSPOSE_SIZE = 128
SWARM_CHANNELS = 2**14
SWARM_CHANNELS_PER_PKT = 8
SWARM_PKTS_PER_BCOUNT = SWARM_CHANNELS/SWARM_CHANNELS_PER_PKT
SWARM_SAMPLES_PER_WINDOW = 2*SWARM_CHANNELS
SWARM_RATE = 2496e6

# R2DBE related constants, should probably be imported from some python
# source in the R2DBE git repo
R2DBE_SAMPLES_PER_WINDOW = 32768
R2DBE_RATE = 4096e6

# CUDA Kernels
mod = SourceModule("""
	#include <pycuda-complex.hpp>  // enable support for complex numbers

	__global__ void fill_padded(int h, pycuda::complex<float> *out, int wo, pycuda::complex<float> *in, int wi){
		// pads with zeros only along x axis

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (y < h){
			if (x < wi){
				out[y*wo+x] = in[y*wi+x];
			} else if (x < wo) {
				out[y*wo+x]._M_re = 0;
				out[y*wo+x]._M_im = 0;
			}
		}
	}

	__global__ void downsample(float *a, int Na, float *b, int Nb, int next_start, int simple_s)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < Nb & tid*simple_s +next_start < Na)
		{
			b[tid] = a[tid*simple_s + next_start];
		}
	}

	__global__ void nearest(float *a, float *b, int Nb, double c){
		/*
		This kernel uses a round-half-to-even tie-breaking rule which is
		opposite that of python's interp_1d.
		*/
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < Nb) {
			b[tid] = a[__double2int_rn(tid*c)];
		}
	}

	texture<float,2,cudaReadModeElementType> a_tex;
	__global__ void copy_texture_kernel(float *b, int Nb, double c, int Nf){
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < Nb){
			double sy = floor(tid * c / Nf);
			double sx = tid * c - sy * Nf;
			b[tid] = tex2D(a_tex,sx+0.5f,sy+0.5f);
		}
	}

	""")

def get_time(dt):
  """ Return datetime object in seconds """
  return dt.days*24*3600 + dt.seconds + dt.microseconds*1e-6

def resample_sdbe_to_r2dbe_zpfft(Xs):
	"""
	Resample SWARM spectrum product in time-domain at R2DBE rate using
	zero-padding and a radix-2 iFFT algorithm.
	
	Arguments:
	----------
	Xs -- MxN numpy array in which the zeroth dimension is increasing
	snapshot index, and the first dimension is the positive frequency
	half of the spectrum.
	
	Returns:
	--------
	xs -- The time-domain signal sampled at the R2DBE rate.
	next_start_vec -- Start indecies for each FFT window.

	"""

	# timestep sizes for SWARM and R2DBE rates
	dt_s = 1.0/SWARM_RATE
	dt_r = 1.0/R2DBE_RATE
	
	# we need to oversample by factor 64 and then undersample by factor 39
	simple_r = 64 # 4096
	simple_s = 39 # 2496
	fft_window_oversample = 2*SWARM_CHANNELS*simple_r # 2* due to real FFT
	
	# oversample timestep size
	dt_f = dt_s/simple_r
	
	# the timespan of one SWARM FFT window
	T_s = dt_s*SWARM_SAMPLES_PER_WINDOW
	
	# what are these...?
	x_t2_0 = None
	x_t2_1 = None
	
	# time vectors over one SWARM FFT window in different step sizes
	t_r = arange(0,T_s,dt_r)
	t_s = arange(0,T_s,dt_s)
	t_f = arange(0,T_s,dt_f)
	
	# offset in oversampled time series that corresponds to one dt_r step
	# from the last R2DBE rate sample in the previous window
	next_start = 0 
	
	# some time offsets...?
	offset_in_window_offset_s = list()
	offset_global_s = list()
	
	# total number of time series samples
	N_x = int(ceil(Xs.shape[0]*SWARM_SAMPLES_PER_WINDOW*dt_s/dt_r))
	# and initialize the output
	xs = zeros(N_x,dtype=float32)
	#fine_sample_index = zeros(N_x)
	next_start_vec = zeros(Xs.shape[0])
	# index in output where samples from next window are stored
	start_output = 0

	# cuFFT plan for complex to real DFT
	plan = cu_fft.Plan(fft_window_oversample,complex64,float32)

	# padding kernel
	fill_padded = mod.get_function("fill_padded")

	# downsampling kernel
	downsample = mod.get_function("downsample")

	# FFT scaling kernel
	scale = ElementwiseKernel(
			"float *a",
			"a[i] = {0} * a[i]".format(1./fft_window_oversample),"scale")

	# max size of resampled chunk from a single window
	xs_chunk_size_max = int32(ceil((1. * fft_window_oversample)/simple_s))

	# create memory on device for cuFFT
       	xf_d = gpuarray.empty(fft_window_oversample,dtype=float32)
	xp_d = gpuarray.zeros(fft_window_oversample/2+1, dtype=complex64)
	y_d = gpuarray.empty(xs_chunk_size_max,dtype=float32)

	for ii in range(Xs.shape[0]):

		# move window to device
		x_d = gpuarray.to_gpu(Xs[ii,:])

		# threads per block
		# number of blocks (keep the array as zeros to save time)
		TPB = 1024
		nB = int(ceil(1. * Xs.shape[1] / TPB))
		# pad with zeros to oversample by 64
		fill_padded(int32(1), xp_d, int32(fft_window_oversample/2+1),\
			    x_d, int32(Xs.shape[1]),\
			    block=(TPB,1,1), grid=(nB,1))

		# iFFT
		cu_fft.ifft(xp_d,xf_d,plan,scale=False)

		xs_chunk_size = int32(ceil((1. * fft_window_oversample - next_start)/simple_s))
		# threads per block
		TPB = 64
		# number of blocks
		nB = ceil(1. * xs_chunk_size / TPB).astype(int)
		## undersample by 39 to correct rate, and start at the correct 
		## offset in this window
		downsample(xf_d,int32(fft_window_oversample),\
				y_d,xs_chunk_size,
				int32(next_start),int32(simple_s),\
				block=(TPB,1,1),grid=(nB,1))

		# rescale from ifft using ElementwiseKernel
		scale(y_d)

		# pull data back onto host
		xs_chunk = y_d.get()

		# fill output numpy array
		stop_output = start_output+xs_chunk_size
		xs[start_output:stop_output] = xs_chunk[:xs_chunk_size]
		# update the starting index in the output array
		start_output = stop_output

		# mark the time of the last used sample relative to the start
		# of this window
		time_window_start_to_last_used_sample = t_f[next_start::39][-1]
		# calculate the remaining time in this window
		time_remaining_in_window = T_s-time_window_start_to_last_used_sample
		# convert to the equivalent number of oversample timesteps
		num_dt_f_steps_short = round(time_remaining_in_window/dt_f)
		next_start_vec[ii] = next_start
		if (num_dt_f_steps_short == 0):
			next_start = 0
		else:
			next_start = simple_s - num_dt_f_steps_short
	return xs,next_start_vec

def resample_sdbe_to_r2dbe_fft_interp(Xs,interp_kind="nearest"):
	"""
	Resample SWARM spectrum product in time-domain at R2DBE rate using
	iFFT and then interpolation in the time-domain.
	
	Arguments:
	----------
	Xs -- MxN numpy array in which the zeroth dimension is increasing
	snapshot index, and the first dimension is the positive frequency
	half of the spectrum.
	interp_kind -- Kind of interpolation.
	
	Returns:
	--------
	xs -- The time-domain signal sampled at the R2DBE rate.
	"""
	# timestep sizes for SWARM and R2DBE rates
	dt_s = 1.0/SWARM_RATE
	dt_r = 1.0/R2DBE_RATE
	
	# cuFFT plan for complex to real DFT
	plan = cu_fft.Plan(SWARM_SAMPLES_PER_WINDOW,complex64,float32,Xs.shape[0])

	# load complex spectrum to device
	x_d = gpuarray.to_gpu(Xs)
	xp_d = gpuarray.empty((Xs.shape[0],Xs.shape[1]+1),dtype=complex64)

	# pad nyquist with zeros
	block = (32,32,1)
	grid = (int(ceil(1. * (Xs.shape[1]+1) / block[1])), int(ceil(1. * Xs.shape[0] / block[0])))
	fill_padded = mod.get_function("fill_padded")
	fill_padded(int32(Xs.shape[0]),xp_d,int32(Xs.shape[1]+1),x_d,int32(Xs.shape[1]),\
		block=block,grid=grid)

	# allocate memory for time series
	xf_d = gpuarray.empty((Xs.shape[0],SWARM_SAMPLES_PER_WINDOW),float32)

	# calculate time series, include scaling
	cu_fft.ifft(xp_d,xf_d,plan,scale=True)

	# and interpolate
	xs_size = int(floor(Xs.shape[0]*SWARM_SAMPLES_PER_WINDOW*dt_s/dt_r)) - 1
	TPB = 64				# threads per block
	nB = int(ceil(1. * xs_size / TPB))	# number of blocks
	xs_d = gpuarray.empty(xs_size,float32)	# decimated time-series 
	if interp_kind == 'nearest':
		# compile kernel
		nearest_interp = mod.get_function(interp_kind)
		# call kernel
		nearest_interp(xf_d,xs_d,int32(xs_size),float64(dt_r/dt_s),block=(TPB,1,1),grid=(nB,1))
	elif interp_kind == 'linear':
		# compile kernel
		linear_interp = mod.get_function("copy_texture_kernel")
		# get texture reference
		a_texref = mod.get_texref("a_tex")
		a_texref.set_filter_mode(drv.filter_mode.LINEAR)	# linear
		#a_texref.set_filter_mode(drv.filter_mode.POINT)	# nearest-neighbor
		# move time series to texture reference
		# following http://lists.tiker.net/pipermail/pycuda/2009-November/001916.html
		descr = drv.ArrayDescriptor()
		descr.format= drv.array_format.FLOAT
		descr.height = Xs.shape[0]
		descr.width = SWARM_SAMPLES_PER_WINDOW
		descr.num_channels = 1
		a_texref.set_address_2d(xf_d.gpudata,descr,SWARM_SAMPLES_PER_WINDOW*4)
		# set up linear interpolation over texture
		linear_interp(xs_d,int32(xs_size),float64(dt_r/dt_s),int32(SWARM_SAMPLES_PER_WINDOW),\
				texrefs=[a_texref],block=(TPB,1,1),grid=(nB,1))

	return xs_d.get()

if __name__ == "__main__":
	"""
	Test script
	"""

	# size of fake spectra
	N_Beng_count = 8
	spectra_shape = (127 * N_Beng_count, 2**14)
	spectra = standard_normal(spectra_shape) + 1j * standard_normal(spectra_shape)

	# nearest-neighor
	tic = get_process_cpu_time()
	xs_nearest = resample_sdbe_to_r2dbe_fft_interp(spectra,interp_kind='nearest')
	toc = get_process_cpu_time()
	time_nearest = toc-tic

	# linear 
	tic = get_process_cpu_time()
	xs_linear = resample_sdbe_to_r2dbe_fft_interp(spectra,interp_kind='linear')
	toc = get_process_cpu_time()
	time_linear = toc-tic

	# zero-padding FFT
	tic = get_process_cpu_time()
	xs_zpfft = resample_sdbe_to_r2dbe_zpfft(spectra)
	toc = get_process_cpu_time()
	time_zpfft = toc-tic

	print 'For {0} B engine frames:'.format(N_Beng_count)
	print 'nearest-neighbor interpolation:',time_nearest
	print 'linear interpolation:',time_linear
	print 'zero-padding FFT:',time_zpfft
