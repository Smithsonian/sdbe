'''
Test different configurations of FFTs.
'''

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.elementwise as el

import scikits.cuda.cufft as cufft

from numpy import complex64,float32,int32,uint32,array,arange,\
			empty,zeros,allclose,ceil,unique,hstack,max,abs
from numpy.random import standard_normal,seed
from numpy.fft import irfft,rfft
import numpy as np

from scipy.signal import resample

BENG_CHANNELS_ = 16384
BENG_CHANNELS = (BENG_CHANNELS_ + 1)
BENG_SNAPSHOTS = 128

SWARM_RATE = 2496e6
R2DBE_RATE = 4096e6

kernel_source = """
#include <cufft.h>
__global__ void zero_out(cufftComplex *a, int32_t n)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n){
    a[tid] = make_cuComplex(0.,0.);
  }
}
"""
###############################################################
def scenario_contiguous_batched39_resample(num_snapshots,tic,toc):
  '''
  # Scenario: Fourier resample of num_snapshots 
  # A iFFT: [num_snapshots,16385] complex64 --> 
  # B FFT: [39,num_snapshots/39 * 32768] float32 --> 
  # C iFFT + zero-padding: [39,num_snapshots/39* 32768*4096/2496/ 2 + 1] complex 64 -->
  # [39,num_snapshots * 32768 * 4096 / 2496] float32
  #
  # 1 C(B(A(gpu_1)))  = C(B(gpu_2)) = C(gpu_1) = gpu_2
  # num_snapshots is a multiple of 39.  
  # A executed using batch = num_snapshots
  # B&C executed using batch = num_snapshots / 39
  '''
  print '\nContiguous channel Fourier resampling scenario in batches of 39:'
  assert num_snapshots % 39 is 0, 'error: num_snapshots must be integer multiple of 39'

  # construct arrays
  batch = num_snapshots / 39
  print 'batch: %d' % batch
  gpu_1 = cuda.mem_alloc(int(8 * batch * (39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)))
  gpu_2 = cuda.mem_alloc(int(4 * batch * (39*2 * BENG_CHANNELS_ * R2DBE_RATE / SWARM_RATE)))
  cpu_in = standard_normal(num_snapshots*BENG_CHANNELS) + 1j * standard_normal(num_snapshots*BENG_CHANNELS)
  cpu_in = cpu_in.astype(complex64)

  # create FFT plans
  n_A = array([2*BENG_CHANNELS_],int32)
  inembed_A = array([BENG_CHANNELS],int32)
  onembed_A = array([2*BENG_CHANNELS_],int32)
  plan_A = cufft.cufftPlanMany(1, n_A.ctypes.data, inembed_A.ctypes.data, 1, BENG_CHANNELS,
  	                                     onembed_A.ctypes.data, 1, 2*BENG_CHANNELS_,
  					     cufft.CUFFT_C2R, num_snapshots)

  n_B = array([39*2*BENG_CHANNELS_],int32)
  inembed_B = array([39*2*BENG_CHANNELS_],int32)
  onembed_B = array([int(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)],int32)
  plan_B = cufft.cufftPlanMany(1, n_B.ctypes.data,
					inembed_B.ctypes.data,1,39*2*BENG_CHANNELS_,
					onembed_B.ctypes.data,1,int32(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1),
					cufft.CUFFT_R2C, batch)

  n_C = array([39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE],int32)
  inembed_C = array([39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1],int32)
  onembed_C = array([39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE],int32)
  plan_C = cufft.cufftPlanMany(1, n_C.ctypes.data,
					inembed_C.ctypes.data,1,int32(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1),
					onembed_C.ctypes.data,1,int32(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE),
					cufft.CUFFT_C2R, batch)

  # zero out gpu_1
  kernel_module = SourceModule(kernel_source)
  zero_out = kernel_module.get_function('zero_out')

  # sanity check:
  zero_out(gpu_1,int32(batch * (39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)),
	block=(1024,1,1),grid=(int(ceil(batch*(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)/1024.)),1))
  cpu_out = empty((batch * (39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)),complex64)
  cuda.memcpy_dtoh(cpu_out,gpu_1)
  assert len(unique(cpu_out)) == 1, 'problem with zero_out'

  # move data to device
  cuda.memcpy_htod(gpu_1,cpu_in)

  tic.record()

  # Turn SWARM snapshots into timeseries
  cufft.cufftExecC2R(plan_A,int(gpu_1),int(gpu_2))

  # zero out gpu_1
  zero_out(gpu_1,int32(batch*(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)),
	block=(1024,1,1),grid=(int(ceil(batch*(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)/1024.)),1))

  # Turn concatenated SWARM time series into single spectrum (already zero-padded)
  cufft.cufftExecR2C(plan_B,int(gpu_2),int(gpu_1))

  # Turn padded SWARM spectrum into time series with R2DBE sampling rate
  cufft.cufftExecC2R(plan_C,int(gpu_1),int(gpu_2))

  toc.record()
  toc.synchronize()

  # check on CPU
  cpu_A = irfft(cpu_in.reshape(num_snapshots,BENG_CHANNELS),axis=-1).astype(float32)
  cpu_B = rfft(cpu_A.reshape(batch,39*2*BENG_CHANNELS_),axis=-1).astype(complex64)
  cpu_C = irfft(hstack([cpu_B, zeros((batch,(39*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE/2+1)-(39*2*BENG_CHANNELS_/2+1)),complex64)]),axis=-1)
  cpu_out = empty(num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE,float32)
  cuda.memcpy_dtoh(cpu_out,gpu_2)

  print 'test results: ', 'pass' if allclose(cpu_C.flatten(),cpu_out/(cpu_C.shape[-1]*2*BENG_CHANNELS_)) else 'fail'
  print 'max residual: ',max(abs(cpu_C.flatten()-cpu_out/(cpu_C.shape[-1]*2*BENG_CHANNELS_)))
  print 'GPU time:', tic.time_till(toc),' ms = ',tic.time_till(toc)/(num_snapshots*0.5*13.128e-3),' x real (both SB)' 

  # destroy plans
  cufft.cufftDestroy(plan_A)
  cufft.cufftDestroy(plan_B)
  cufft.cufftDestroy(plan_C)

###############################################################
def scenario_contiguous_resample(num_snapshots,tic,toc):
  '''
  # Scenario: Fourier resample of num_snapshots 
  # A iFFT: [num_snapshots,16385] complex64 --> 
  # B FFT: [num_snapshots * 32768] float32 --> 
  # C iFFT + zero-padding: [(num_snapshots* 32768 * 4096 / 2496  / 2  + 1] complex 64 -->
  # [num_snapshots * 32768 * 4096 / 2496] float32
  #
  # 1 C(B(A(gpu_1)))  = C(B(gpu_2)) = C(gpu_1) = gpu_2
  # num_snapshots is a multiple of 39.  
  '''
  print '\nContiguous channel Fourier resampling scenario:'

  assert num_snapshots % 39 is 0, 'error: num_snapshots must be integer multiple of 39'

  # construct arrays
  gpu_1 = cuda.mem_alloc(int(4 * (num_snapshots * 2 * BENG_CHANNELS_ * R2DBE_RATE / SWARM_RATE + 2)))
  gpu_2 = cuda.mem_alloc(int(4 * num_snapshots *2 * BENG_CHANNELS_ * R2DBE_RATE / SWARM_RATE))
  cpu_in = standard_normal(num_snapshots*BENG_CHANNELS) + 1j * standard_normal(num_snapshots*BENG_CHANNELS)
  cpu_in = cpu_in.astype(complex64)

  # create FFT plans
  n_A = array([2*BENG_CHANNELS_],int32)
  inembed_A = array([BENG_CHANNELS],int32)
  onembed_A = array([2*BENG_CHANNELS_],int32)
  plan_A = cufft.cufftPlanMany(1, n_A.ctypes.data, inembed_A.ctypes.data, 1, BENG_CHANNELS,
  	                                     onembed_A.ctypes.data, 1, 2*BENG_CHANNELS_,
  					     cufft.CUFFT_C2R, num_snapshots)

  n_B = array([num_snapshots*2*BENG_CHANNELS_],int32)
  inembed_B = array([num_snapshots*2*BENG_CHANNELS_],int32)
  onembed_B = array([num_snapshots*2*BENG_CHANNELS_/2+1],int32)
  plan_B = cufft.cufftPlanMany(1, n_B.ctypes.data,
					inembed_B.ctypes.data,1,num_snapshots*2*BENG_CHANNELS_,
					onembed_B.ctypes.data,1,num_snapshots*2*BENG_CHANNELS_/2+1,
					cufft.CUFFT_R2C, 1)

  n_C = array([num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE],int32)
  inembed_C = array([num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE/2+1],int32)
  onembed_C = array([num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE],int32)
  plan_C = cufft.cufftPlanMany(1, n_C.ctypes.data,
					inembed_C.ctypes.data,1,int(num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE/2+1),
					onembed_C.ctypes.data,1,int(num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE),
					cufft.CUFFT_C2R, 1)

  # zero out gpu_1
  kernel_module = SourceModule(kernel_source)
  zero_out = kernel_module.get_function('zero_out')

  # sanity check:
  zero_out(gpu_1,int32(num_snapshots * 2 * BENG_CHANNELS_ * R2DBE_RATE / SWARM_RATE / 2 + 1),
	block=(1024,1,1),grid=(int(ceil((num_snapshots*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)/1024.)),1))
  cpu_out = empty((num_snapshots * 2 * BENG_CHANNELS_ * R2DBE_RATE / SWARM_RATE/2 + 1),complex64)
  cuda.memcpy_dtoh(cpu_out,gpu_1)
  assert len(unique(cpu_out)) == 1, 'problem with zero_out'

  # move data to device
  cuda.memcpy_htod(gpu_1,cpu_in)

  tic.record()

  # Turn SWARM snapshots into timeseries
  cufft.cufftExecC2R(plan_A,int(gpu_1),int(gpu_2))

  # zero out gpu_1
  zero_out(gpu_1,int32(num_snapshots * 2 * BENG_CHANNELS_ * R2DBE_RATE / SWARM_RATE / 2 + 1),
	block=(1024,1,1),grid=(int(ceil((num_snapshots*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE/2+1)/1024.)),1))

  # Turn concatenated SWARM time series into single spectrum (already zero-padded)
  cufft.cufftExecR2C(plan_B,int(gpu_2),int(gpu_1))

  # Turn padded SWARM spectrum into time series with R2DBE sampling rate
  cufft.cufftExecC2R(plan_C,int(gpu_1),int(gpu_2))

  toc.record()
  toc.synchronize()

  # check on CPU
  cpu_A = irfft(cpu_in.reshape(num_snapshots,BENG_CHANNELS),axis=-1).astype(float32)
  cpu_B = rfft(cpu_A.flatten()).astype(complex64)
  cpu_C = irfft(hstack([cpu_B, zeros((num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE/2+1)-(num_snapshots*2*BENG_CHANNELS_/2+1),complex64)]))
  cpu_out = empty(num_snapshots*2*BENG_CHANNELS_* R2DBE_RATE/SWARM_RATE,float32)
  cuda.memcpy_dtoh(cpu_out,gpu_2)

  print 'test results: ', 'pass' if allclose(cpu_C,cpu_out/(cpu_out.size*2*BENG_CHANNELS_)) else 'fail'
  print 'max residual:', max(abs(cpu_C-cpu_out/(cpu_out.size*2*BENG_CHANNELS_)))
  print 'real time:', batch * 13.128e-3,' ms'
  print 'GPU time:', tic.time_till(toc),' ms = ',tic.time_till(toc)/(num_snapshots*0.5*13.128e-3),' x real (both SB)' 
  
  # destroy plans
  cufft.cufftDestroy(plan_A)
  cufft.cufftDestroy(plan_B)
  cufft.cufftDestroy(plan_C)

###############################################################
def scenario_contiguous_channels(batch,tic,toc):
  '''
  # Scenario: batched IFFT of batch snapshots
  # batch x 16385 complex64 --> batch x 32768 float32
  # no padding.
  '''
  
  n = array([2*BENG_CHANNELS_],int32)
  seed(12740)
  
  # create batched FFT plan configuration
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  plan = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, BENG_CHANNELS,
  	                                     onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
  					     cufft.CUFFT_C2R, batch)
  
  # construct arrays 
  gpu_in  = cuda.mem_alloc(8*batch*BENG_CHANNELS)		# complex64
  gpu_out = cuda.mem_alloc(4*batch*2*BENG_CHANNELS_)	# float32
  cpu_in = standard_normal(batch*BENG_CHANNELS) + 1j * standard_normal(batch*BENG_CHANNELS)
  cpu_in = cpu_in.astype(complex64)
  cuda.memcpy_htod(gpu_in,cpu_in)
  
  # execute plan
  tic.record()
  cufft.cufftExecC2R(plan,int(gpu_in),int(gpu_out))
  toc.record()
  toc.synchronize()
  
  # read out result
  cpu_out = empty(batch*2*BENG_CHANNELS_,float32)
  cuda.memcpy_dtoh(cpu_out,gpu_out)
  cpu_out.resize((batch,2*BENG_CHANNELS_))
  
  # execute on CPU
  cpu = irfft(cpu_in.reshape((batch,BENG_CHANNELS)),axis=-1)
  
  # destroy plan
  cufft.cufftDestroy(plan)
  
  # test
  print '\nContiguous Channel Scenario:'
  print '1-D %d-element C2R iFFT in batch of %d.' % (n, batch)
  print 'test results: ' 'pass' if allclose(cpu,cpu_out/(2*BENG_CHANNELS_)) else 'fail'
  print 'real time:', batch * 13.128e-3,' ms'
  print 'GPU time:', tic.time_till(toc),' ms =  ',tic.time_till(toc)/(batch*0.5*13.128e-3),' x real (both SB)' 

###############################################################
def scenario_contiguous_channels_oversampled64(batch,tic,toc):
  '''
  Scenario: batched IFFT of 2*2**14*64 channels
  '''
  fft_window_oversample = 64*2*2**14
  n = array([fft_window_oversample],int32)
  
  # create batched FFT plan configuration
  inembed = array([fft_window_oversample/2+1],int32)
  onembed = array([fft_window_oversample],int32)
  plan = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, fft_window_oversample/2+1,
  	                                     onembed.ctypes.data, 1, fft_window_oversample,
  					     cufft.CUFFT_C2R, batch)
  # construct arrays 
  gpu_in  = cuda.mem_alloc(8*batch*(fft_window_oversample/2+1))	# complex64
  gpu_out = cuda.mem_alloc(4*batch*fft_window_oversample)	# float32
  data_shape = (batch,fft_window_oversample/2+1)
  cpu_in = standard_normal(data_shape) + 1j * standard_normal(data_shape)
  cpu_in = cpu_in.astype(complex64)
  cuda.memcpy_htod(gpu_in,cpu_in)
  # execute plan
  
  tic.record()
  cufft.cufftExecC2R(plan,int(gpu_in),int(gpu_out))
  toc.record()
  toc.synchronize()
  
  # read out result
  cpu_out = empty((batch,fft_window_oversample),float32)
  cuda.memcpy_dtoh(cpu_out,gpu_out)
  
  # execute on CPU
  cpu = irfft(cpu_in,axis=-1)
  
  # destroy plan
  cufft.cufftDestroy(plan)
  
  # test
  print '\nOversampling by x64 Scenario with batches:'
  print '1-D %d-element C2R iFFT in batch of %d.' % (n, batch)
  print 'test results: ' 'pass' if allclose(cpu,cpu_out/(fft_window_oversample)) else 'fail'
  print 'real time:', batch * 13.128e-3,' ms'
  print 'GPU time:', tic.time_till(toc),' ms =  ',tic.time_till(toc)/(batch*0.5*13.128e-3),' x real (both SB)' 
###############################################################
def scenario_contiguous_channels_wpadding(batch,tic,toc):
  '''
  # Scenario: batched IFFT of 39 snapshots
  # 39 x 16385 complex64 --> 39 x 32768 float32
  # padding complex input so channel dimension has 16400 elements.  
  '''
  
  n = array([2*BENG_CHANNELS_],int32)
  beng_channels_padded = 16400
  
  # create batched FFT plan configuration
  inembed = array([beng_channels_padded],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  istride = int32(beng_channels_padded)
  plan = cufft.cufftPlanMany(int32(1), n.ctypes.data, inembed.ctypes.data, int32(1), istride,
  	                                     onembed.ctypes.data, int32(1), int32(2*BENG_CHANNELS_),
  					     cufft.CUFFT_C2R, int32(batch))
  # construct arrays 
  gpu_in  = cuda.mem_alloc(8*batch*beng_channels_padded)	# complex64
  gpu_out = cuda.mem_alloc(4*batch*2*BENG_CHANNELS_)	# float32
  cpu_in = standard_normal(batch*beng_channels_padded) + 1j * standard_normal(batch*beng_channels_padded)
  cpu_in = cpu_in.astype(complex64)
  cuda.memcpy_htod(gpu_in,cpu_in)
  # execute plan
  
  tic.record()
  cufft.cufftExecC2R(plan,int(gpu_in),int(gpu_out))
  toc.record()
  toc.synchronize()
  
  # read out result
  cpu_out = empty(batch*2*BENG_CHANNELS_,float32)
  cuda.memcpy_dtoh(cpu_out,gpu_out)
  cpu_out.resize((batch,2*BENG_CHANNELS_))
  
  # execute on CPU
  cpu = irfft(cpu_in.reshape((batch,beng_channels_padded))[:,:BENG_CHANNELS],axis=-1)
  
  # destroy plan
  cufft.cufftDestroy(plan)
  
  # test
  print '\nContiguous Channel w/ Padding Scenario:'
  print '1-D %d-element C2R iFFT in batch of %d.' % (n, batch)
  print 'test results: ' 'pass' if allclose(cpu,cpu_out/(2*BENG_CHANNELS_)) else 'fail'
  print 'real time:', batch * 13.128e-3,' ms'
  print 'GPU time:', tic.time_till(toc),' ms =  ',tic.time_till(toc)/(batch*0.5*13.128e-3),' x real (both SB)' 
  
###############################################################
def scenario_contiguous_snapshots(batch,tic,toc):
  '''
  # Scenario: batched IFFT of 39 snapshots
  # 16385 x 39 complex64 --> 39 x 32768 float32
  '''
  
  n = array([2*BENG_CHANNELS_],int32)
  seed(12740)
  
  # create batched FFT plan configuration
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  plan = cufft.cufftPlanMany(int32(1), n.ctypes.data,
  				inembed.ctypes.data, int32(batch), int32(1),
  	                        onembed.ctypes.data, int32(1), int32(2*BENG_CHANNELS_),
  			    cufft.CUFFT_C2R, int32(batch))
  # construct arrays 
  gpu_in  = cuda.mem_alloc(8*batch*BENG_CHANNELS)	# complex64
  gpu_out = cuda.mem_alloc(4*batch*2*BENG_CHANNELS_)	# float32
  cpu_in = standard_normal(batch*BENG_CHANNELS) + 1j * standard_normal(batch*BENG_CHANNELS)
  cpu_in = cpu_in.astype(complex64)
  cuda.memcpy_htod(gpu_in,cpu_in)
  
  # execute plan
  tic.record()
  cufft.cufftExecC2R(plan,int(gpu_in),int(gpu_out))
  toc.record()
  toc.synchronize()
  
  # read out result
  cpu_out = empty(batch*2*BENG_CHANNELS_,float32)
  cuda.memcpy_dtoh(cpu_out,gpu_out)
  cpu_out.resize((batch,2*BENG_CHANNELS_))
  
  # execute on CPU
  cpu = irfft(cpu_in.reshape((batch,BENG_CHANNELS),order='F'),axis=-1)
  
  # destroy plan
  cufft.cufftDestroy(plan)
  
  # test
  print '\nContiguous Snapshot Scenario:'
  print '1-D %d-element C2R iFFT in batch of %d.' % (n, batch)
  print 'test results: ' 'pass' if allclose(cpu,cpu_out/(2*BENG_CHANNELS_)) else 'fail'
  print 'real time:', batch * 13.128e-3,' ms'
  print 'GPU time:', tic.time_till(toc),' ms =  ',tic.time_till(toc)/(batch*0.5*13.128e-3),' x real (both SB)' 

################################################################################################
# two timers for speed-testing
tic = cuda.Event()
toc = cuda.Event()

batch = 9 * 39
#batch = 39
print '\nTiming scenarios are for a single side-band'

'''
Notes:
- Best performance I see for batched iFFT (n > 128) is  0.75  x real for both sidebands.
- I see no gain from padding input.
- 39 B-frames requires too much memory
- Best performace for Fourier resamping (not batched) is ~4.2x real for both sidebands.
'''

scenario_contiguous_channels(batch,tic,toc)
#scenario_contiguous_resample(batch,tic,toc)
scenario_contiguous_batched39_resample(batch,tic,toc)
#scenario_contiguous_channels_oversampled64(batch,tic,toc)
num_snapshots=batch

print '\ndone\n'

