'''
Test different configurations of FFTs.
'''

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.elementwise as el

import scikits.cuda.cufft as cufft

from numpy import complex64,float32,int32,uint32,array,arange,empty,zeros,allclose
from numpy.random import standard_normal,seed
from numpy.fft import irfft
import numpy as np

BENG_CHANNELS_ = 16384
BENG_CHANNELS = (BENG_CHANNELS_ + 1)
BENG_SNAPSHOTS = 128

#scale_inplace = el.ElementwiseKernel(
#                        "float *x, float a",
#                        "x[i] = a * x[i]","scale")



###############################################################
def scenario_contiguous_channels(batch,tic,toc):
  # Scenario: batched IFFT of 39 snapshots
  # 39 x 16385 complex64 --> 39 x 32768 float32
  # no padding.
  
  n = array([2*BENG_CHANNELS_],int32)
  seed(12740)
  
  # create batched FFT plan configuration
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  plan = cufft.cufftPlanMany(int32(1), n.ctypes.data, inembed.ctypes.data, int32(1), int32(BENG_CHANNELS),
  	                                     onembed.ctypes.data, int32(1), int32(2*BENG_CHANNELS_),
  					     cufft.CUFFT_C2R, int32(batch))
  #cufft.cufftSetCompatibilityMode(plan,0x01)
  
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
  
def scenario_contiguous_channels_wpadding(batch,tic,toc):
  # Scenario: batched IFFT of 39 snapshots
  ###############################################################
  # Scenario: batched IFFT of 39 snapshots
  # 39 x 16385 complex64 --> 39 x 32768 float32
  # padding complex input so channel dimension has 16400 elements.  
  
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
  
def scenario_contiguous_snapshots(batch,tic,toc):
  ###############################################################
  # Scenario: batched IFFT of 39 snapshots
  # 16385 x 39 complex64 --> 39 x 32768 float32
  
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

batch = 128 * 4
print '\nTiming scenarios are for a single side-band'

'''
Notes:
- Best preformance I see for batched iFFT (n > 128) is  0.75  x real for both sidebands.
- I see no gain from padding input.
'''

scenario_contiguous_channels(batch,tic,toc)
