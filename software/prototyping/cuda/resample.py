'''
Various resampling options
'''

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import scikits.cuda.cufft as cufft

from numpy import complex64,float32,int32,uint32,array,arange,floor,float64,\
			empty,zeros,allclose,ceil,unique,hstack,max,abs,median,\
			rint,all
from numpy.random import standard_normal,seed
from numpy.fft import irfft,rfft
import numpy as np

from scipy.interpolate import interp1d

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

__global__ void nearest(float *a, float *b, int Nb, double c, float d){
  /*
  This kernel uses a round-half-to-even tie-breaking rule which is
  opposite that of python's interp_1d.
  a: input_array
  b: output_array
  Nb: size of array b
  c: stride for interpolation: b[i] = d*a[int(c*i)]
  */
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < Nb) {
    b[tid] = d*a[__double2int_rn(tid*c)];
  }
}

__global__ void linear(const float *a, float *b, const int Nb, const double c, const float d){
 /*
  a: input_array
  b: output_array
  Nb: size of array b
  c: conversion factor between a and b indices. 
  Note: type conversions are slowing this down.
  */
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < Nb) {
    int ida = __double2int_rd(tid*c); // round down
    b[tid] = d * ( a[ida]*(1.-(c*tid-ida)) + a[ida+1]*(c*tid-ida) );
  }
}

"""

kernel_source_1 = """
#include <cufft.h>

__global__ void linear(const float *a, float *b, const int N, const float *wgt, const int *ida){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N){
    int i = ida[tid];
    float w = wgt[tid];
    b[tid] = a[i]*(1.f-w) + a[i+1]*w;
  }
}

__global__ void linear1(const float *a, float *b, const int N, const float *wgt, const int *ida){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = ida[tid];
  float w = wgt[tid];
  // loop over groups of 39 snapshots (assuming we have 128 such chunks)
  for (int ichunk=0; ichunk<128; ichunk += 1){
    //b[ichunk*2097152+tid] = a[i+1277952*ichunk]*(1.f-w) + a[i+1+1277952*ichunk]*w;
    if (i+1+1277952*ichunk<N){
      b[ichunk*2097152+tid] = a[i+1277952*ichunk]*(1.f-w) + a[i+1+1277952*ichunk]*w;
    } else {
      b[ichunk*2097152+tid] = a[i+1277952*ichunk];
    }
  }
}
"""

def fft_batched(gpu_1,gpu_2,num_snapshots,snapshots_per_batch=39,cpu_check=True):
  '''
  gpu_1: pointer to Mx16385 array on GPU device where zeroth dimension is positive frequency half of spectrum
        and the first dimension is is increasing snapshot index.  This array will be destroyed.
	Must have byte size: int(8*batch_size*(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1))
  gpu_2: pointer to result
  snapshots_per_batch: number of snapshots grouped for resampling (% 39 == 0)
  '''
  tic = cuda.Event()
  toc = cuda.Event()

  batch_size = num_snapshots / snapshots_per_batch
  print 'batch size: %d' % batch_size

  # create FFT plans
  n_A = array([2*BENG_CHANNELS_],int32)
  inembed_A = array([BENG_CHANNELS],int32)
  onembed_A = array([2*BENG_CHANNELS_],int32)
  plan_A = cufft.cufftPlanMany(1, n_A.ctypes.data, inembed_A.ctypes.data, 1, BENG_CHANNELS,
  	                                     onembed_A.ctypes.data, 1, 2*BENG_CHANNELS_,
  					     cufft.CUFFT_C2R, num_snapshots)

  n_B = array([snapshots_per_batch*2*BENG_CHANNELS_],int32)
  inembed_B = array([snapshots_per_batch*2*BENG_CHANNELS_],int32)
  onembed_B = array([int(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1)],int32)
  plan_B = cufft.cufftPlanMany(1, n_B.ctypes.data,
			inembed_B.ctypes.data,1,snapshots_per_batch*2*BENG_CHANNELS_,
			onembed_B.ctypes.data,1,int32(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1),
					cufft.CUFFT_R2C, batch_size)

  n_C = array([snapshots_per_batch*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE],int32)
  inembed_C = array([snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1],int32)
  onembed_C = array([snapshots_per_batch*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE],int32)
  plan_C = cufft.cufftPlanMany(1, n_C.ctypes.data,
			inembed_C.ctypes.data,1,int32(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1),
			onembed_C.ctypes.data,1,int32(snapshots_per_batch*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE),
					cufft.CUFFT_C2R, batch_size)

  # fetch kernel that zeroes out an array
  kernel_module = SourceModule(kernel_source)
  zero_out = kernel_module.get_function('zero_out')

  tic.record()

  # Turn SWARM snapshots into timeseries
  cufft.cufftExecC2R(plan_A,int(gpu_1),int(gpu_2))

  # zero out gpu_1
  zero_out(gpu_1,int32(batch_size*(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1)),
	block=(1024,1,1),
	grid=(int(ceil(batch_size*(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1)/1024.)),1))

  # Turn concatenated SWARM time series into single spectrum (already zero-padded)
  cufft.cufftExecR2C(plan_B,int(gpu_2),int(gpu_1))

  # Turn padded SWARM spectrum into time series with R2DBE sampling rate
  cufft.cufftExecC2R(plan_C,int(gpu_1),int(gpu_2))

  toc.record()
  toc.synchronize()

  # check on CPU
  if (cpu_check):
    cpu_A = irfft(cpu_in.reshape(num_snapshots,BENG_CHANNELS),axis=-1).astype(float32)
    cpu_B = rfft(cpu_A.reshape(batch_size,snapshots_per_batch*2*BENG_CHANNELS_),axis=-1).astype(complex64)
    cpu_C = irfft(hstack([cpu_B, 
			zeros((batch_size,(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1)-
				(snapshots_per_batch*BENG_CHANNELS_+1)),complex64)]),axis=-1)
    cpu_out = empty(num_snapshots*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE,float32)
    cuda.memcpy_dtoh(cpu_out,gpu_2)

    print 'test results: ', 'pass' if allclose(cpu_C.flatten(),cpu_out/(cpu_C.shape[-1]*2*BENG_CHANNELS_)) else 'fail'
    print 'max residual: ',max(abs(cpu_C.flatten()-cpu_out/(cpu_C.shape[-1]*2*BENG_CHANNELS_)))

  print 'GPU time:', tic.time_till(toc),' ms = ',tic.time_till(toc)/(num_snapshots*0.5*13.128e-3),' x real (both SB)' 

  # destroy plans
  cufft.cufftDestroy(plan_A)
  cufft.cufftDestroy(plan_B)
  cufft.cufftDestroy(plan_C)

##################################################################################

def fft_interp(gpu_1,gpu_2,num_snapshots,interp_kind='nearest',cpu_check=True):
  '''
  Batched fft to time series and then interpolation to resample.
  No filter applied yet...
  '''
  tic = cuda.Event()
  toc = cuda.Event()

  batch_size = num_snapshots
  print 'batch size: %d' % batch_size

  # create batched FFT plan configuration
  n = array([2*BENG_CHANNELS_],int32)
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  plan = cufft.cufftPlanMany(1, n.ctypes.data,
 			inembed.ctypes.data, 1, BENG_CHANNELS,
                        onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
  			cufft.CUFFT_C2R, batch_size)

  # fetch kernel that resamples 
  kernel_module = SourceModule(kernel_source)
  interp_1d = kernel_module.get_function(interp_kind)

  # execute plan
  cufft.cufftExecC2R(plan,int(gpu_1),int(gpu_2))

  # interpolate
  tic.record()
  xs_size = int(floor(batch_size*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)) - 1
  TPB = 512                               # threads per block
  nB = int(ceil(1. * xs_size / TPB))      # number of blocks
  if interp_kind is 'linear':
    interp_1d(gpu_2,gpu_1,int32(xs_size),float64(SWARM_RATE/R2DBE_RATE),float32(1./(2*BENG_CHANNELS_)),
		block=(TPB,1,1),grid=(nB,1))
  else:
    interp_1d(gpu_2,gpu_1,int32(xs_size),float64(SWARM_RATE/R2DBE_RATE),float32(1./(2*BENG_CHANNELS_)),
		block=(TPB,1,1),grid=(nB,1))

  toc.record()
  toc.synchronize()

  print 'GPU time:', tic.time_till(toc),' ms = ',tic.time_till(toc)/(num_snapshots*0.5*13.128e-3),' x real (both SB)' 

  # destroy plan
  cufft.cufftDestroy(plan)

  # check on CPU
  if (cpu_check):
    # timestep sizes for SWARM and R2DBE rates
    dt_s = 1.0/SWARM_RATE
    dt_r = 1.0/R2DBE_RATE
    # the timespan of one SWARM FFT window
    T_s = dt_s*2*BENG_CHANNELS_
    # the timespan of all SWARM data
    T_s_all = T_s*batch_size
    # get time-domain signal
    xs_swarm_rate = irfft(cpu_in,n=2*BENG_CHANNELS_,axis=1).flatten()
    # and calculate sample points
    t_swarm_rate = arange(0,T_s_all,dt_s)
    print t_swarm_rate[0],t_swarm_rate[-1]
    # calculate resample points (subtract one dt_s from end to avoid extrapolation)
    t_r2dbe_rate = arange(0,T_s_all-dt_s,dt_r)
    # and interpolate
    x_interp = interp1d(t_swarm_rate,xs_swarm_rate,kind=interp_kind)
    cpu_A = x_interp(t_r2dbe_rate).astype(float32)

    cpu_out = np.empty_like(cpu_A,dtype=float32)
    cuda.memcpy_dtoh(cpu_out,gpu_1)

    print 'median residual: ',median(abs(cpu_A-cpu_out))
    if interp_kind is 'nearest':
      cpu_A[::32] = 0
      cpu_out[::32] = 0
    print 'test results: ', 'pass' if allclose(cpu_A,cpu_out) else 'fail'
##################################################################################

def fft_interp_1(gpu_1,gpu_2,num_snapshots,interp_kind='linear',cpu_check=True):
  '''
  Batched fft to time series and then interpolation to resample.
  Use devuce memory to store coefficients and index values.
  '''
  tic = cuda.Event()
  toc = cuda.Event()

  batch_size = num_snapshots // 39
  print 'batch size: %d' % batch_size

  # create batched FFT plan configuration
  n = array([2*BENG_CHANNELS_],int32)
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  plan = cufft.cufftPlanMany(1, n.ctypes.data,
 			inembed.ctypes.data, 1, BENG_CHANNELS,
                        onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
  			cufft.CUFFT_C2R, batch_size)

  # pre-calculate coefficients on CPU (c*tid-ida)
  r2dbe_samples = int(num_snapshots*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)
  ctid = SWARM_RATE/R2DBE_RATE*arange(r2dbe_samples)
  cpu_ida  = floor(SWARM_RATE/R2DBE_RATE*arange(r2dbe_samples)).astype(int32)
  cpu_wgt  = (ctid-cpu_ida).astype(float32)

  gpu_ida = cuda.mem_alloc(4 * r2dbe_samples)
  cuda.memcpy_htod(gpu_ida,cpu_ida)
  gpu_wgt = cuda.mem_alloc(4 * r2dbe_samples)
  cuda.memcpy_htod(gpu_wgt,cpu_wgt)

  kernel_module = SourceModule(kernel_source_1)
  linear = kernel_module.get_function('linear')

  # execute plan
  for ib in range(num_snapshots / batch_size):
    cufft.cufftExecC2R(plan,
			int(gpu_1)+(8*ib*batch_size*BENG_CHANNELS),
			int(gpu_2)+(4*ib*batch_size*2*BENG_CHANNELS_))

  # resample
  tic.record()
  linear(gpu_2,gpu_1,int32(r2dbe_samples),gpu_wgt,gpu_ida,
	block=(512,1,1),grid=(r2dbe_samples/512,1))
  toc.record()
  toc.synchronize()

  print 'GPU time:', tic.time_till(toc),' ms = ',tic.time_till(toc)/(num_snapshots*0.5*13.128e-3),' x real (both SB)' 

  # destroy plan
  cufft.cufftDestroy(plan)

  # check on CPU
  if (cpu_check):
    # timestep sizes for SWARM and R2DBE rates
    dt_s = 1.0/SWARM_RATE
    dt_r = 1.0/R2DBE_RATE
    # the timespan of one SWARM FFT window
    T_s = dt_s*2*BENG_CHANNELS_
    # the timespan of all SWARM data
    T_s_all = T_s*num_snapshots
    # get time-domain signal
    xs_swarm_rate = irfft(cpu_in,n=2*BENG_CHANNELS_,axis=1).flatten()
    # and calculate sample points
    t_swarm_rate = arange(0,T_s_all,dt_s)
    print t_swarm_rate[0],t_swarm_rate[-1]
    # calculate resample points (subtract one dt_s from end to avoid extrapolation)
    t_r2dbe_rate = arange(0,T_s_all-dt_s,dt_r)
    # and interpolate
    x_interp = interp1d(t_swarm_rate,xs_swarm_rate,kind=interp_kind)
    cpu_A = x_interp(t_r2dbe_rate)

    cpu_out = np.empty_like(cpu_A,dtype=float32)
    cuda.memcpy_dtoh(cpu_out,gpu_1)
    cpu_out /= (2*BENG_CHANNELS_)

    print 'median residual: ',median(abs(cpu_A-cpu_out))
    if interp_kind is 'nearest':
      cpu_A[::32] = 0
      cpu_out[::32] = 0
    print 'test results: ', 'pass' if allclose(cpu_A,cpu_out) else 'fail'

##################################################################################

def fft_interp_2(gpu_1,gpu_2,num_snapshots,interp_kind='linear',cpu_check=True):
  '''
  Batched fft to time series and then interpolation to resample.
  Use devuce memory to store coefficients and index values.
  '''
  tic = cuda.Event()
  toc = cuda.Event()

  batch_size = num_snapshots // 39
  print 'batch size: %d' % batch_size

  # create batched FFT plan configuration
  n = array([2*BENG_CHANNELS_],int32)
  inembed = array([BENG_CHANNELS],int32)
  onembed = array([2*BENG_CHANNELS_],int32)
  plan = cufft.cufftPlanMany(1, n.ctypes.data,
 			inembed.ctypes.data, 1, BENG_CHANNELS,
                        onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
  			cufft.CUFFT_C2R, batch_size)

  # pre-calculate coefficients on CPU (c*tid-ida)
  stride = 39*2*BENG_CHANNELS_
  template_size = int(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)
  ctid = SWARM_RATE/R2DBE_RATE*arange(template_size)
  cpu_ida  = floor(SWARM_RATE/R2DBE_RATE*arange(template_size)).astype(int32)
  cpu_wgt  = (ctid-cpu_ida).astype(float32)

  gpu_ida = cuda.mem_alloc(4 * template_size)
  cuda.memcpy_htod(gpu_ida,cpu_ida)
  gpu_wgt = cuda.mem_alloc(4 * template_size)
  cuda.memcpy_htod(gpu_wgt,cpu_wgt)

  kernel_module = SourceModule(kernel_source_1)
  linear = kernel_module.get_function('linear1')

  # execute plan
  for ib in range(num_snapshots / batch_size):
    cufft.cufftExecC2R(plan,
			int(gpu_1)+(8*ib*batch_size*BENG_CHANNELS),
			int(gpu_2)+(4*ib*batch_size*2*BENG_CHANNELS_))

  # resample
  tic.record()
  linear(gpu_2,gpu_1,int32(num_snapshots*2*BENG_CHANNELS_),gpu_wgt,gpu_ida,
	block=(512,1,1),grid=(2097152/512,1))
  toc.record()
  toc.synchronize()

  print 'GPU time:', tic.time_till(toc),' ms = ',tic.time_till(toc)/(num_snapshots*0.5*13.128e-3),' x real (both SB)' 

  # destroy plan
  cufft.cufftDestroy(plan)

  # check on CPU
  if (cpu_check):
    # timestep sizes for SWARM and R2DBE rates
    dt_s = 1.0/SWARM_RATE
    dt_r = 1.0/R2DBE_RATE
    # the timespan of one SWARM FFT window
    T_s = dt_s*2*BENG_CHANNELS_
    # the timespan of all SWARM data
    T_s_all = T_s*num_snapshots
    # get time-domain signal
    xs_swarm_rate = irfft(cpu_in,n=2*BENG_CHANNELS_,axis=1).flatten()
    # and calculate sample points
    t_swarm_rate = arange(0,T_s_all,dt_s)
    # calculate resample points (subtract one dt_s from end to avoid extrapolation)
    t_r2dbe_rate = arange(0,T_s_all-dt_s,dt_r)
    # and interpolate
    x_interp = interp1d(t_swarm_rate,xs_swarm_rate,kind=interp_kind)
    cpu_A = x_interp(t_r2dbe_rate)

    cpu_out = np.empty_like(cpu_A,dtype=float32)
    cuda.memcpy_dtoh(cpu_out,gpu_1)
    cpu_out /= (2*BENG_CHANNELS_)

    print 'median residual: ',median(abs(cpu_A-cpu_out))
    if interp_kind is 'nearest':
      cpu_A[::32] = 0
      cpu_out[::32] = 0
    print 'test results: ', 'pass' if allclose(cpu_A,cpu_out) else 'fail'


##################################################################################

# mock data
num_snapshots = 39*128
data_shape = (num_snapshots,BENG_CHANNELS)
cpu_in = standard_normal(data_shape) + 1j * standard_normal(data_shape)
cpu_in = cpu_in.astype(complex64)

# batched fft resample: 
if False:
  # move data to device
  snapshots_per_batch = 39
  batch_size = num_snapshots / snapshots_per_batch
  gpu_1 = cuda.mem_alloc(int(8*batch_size*(snapshots_per_batch*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE+1)))
  gpu_2 = cuda.mem_alloc(int(4*batch_size*(snapshots_per_batch*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)))
  cuda.memcpy_htod(gpu_1,cpu_in)
  fft_batched(gpu_1,gpu_2,num_snapshots,snapshots_per_batch=snapshots_per_batch,cpu_check=True)
  gpu_1.free()
  gpu_2.free()

# nearest-neighbor:
if False:
  gpu_1 = cuda.mem_alloc(4*(int(floor(num_snapshots*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)) - 1))
  gpu_2 = cuda.mem_alloc(int(4*num_snapshots*(2*BENG_CHANNELS_)))
  cuda.memcpy_htod(gpu_1,cpu_in)
  fft_interp(gpu_1,gpu_2,num_snapshots,interp_kind='nearest',cpu_check=True)

# linear 
if False:
  gpu_1 = cuda.mem_alloc(4*(int(floor(num_snapshots*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)) - 1))
  gpu_2 = cuda.mem_alloc(4*int(num_snapshots*(2*BENG_CHANNELS_)))
  cuda.memcpy_htod(gpu_1,cpu_in)
  fft_interp(gpu_1,gpu_2,num_snapshots,interp_kind='linear',cpu_check=True)

# linear saving coefficients
if True:
  gpu_1 = cuda.mem_alloc(4*int(num_snapshots*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE))
  gpu_2 = cuda.mem_alloc(4*int(num_snapshots*(2*BENG_CHANNELS_)))
  cuda.memcpy_htod(gpu_1,cpu_in)
  fft_interp_2(gpu_1,gpu_2,num_snapshots,interp_kind='linear',cpu_check=True)
