'''
Module for using cuda kernels in python
'''

import sys
sdbe_scripts_dir = '/home/krosenfe/sdbe/software/prototyping'
sys.path.append(sdbe_scripts_dir)

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import scikits.cuda.cufft as cufft
from kernel_template import kernel_template

from numpy.fft import rfft, irfft
from numpy import complex64,float32,float64,int32,uint32,array,arange,empty,zeros,ceil,roll,hstack
from numpy import min,floor,ceil,exp,angle,reshape,sqrt,roll,log2
from scipy.special import erfinv
from argparse import ArgumentParser
from struct import unpack
import logging
from timeit import default_timer as timer
from collections import defaultdict

import read_r2dbe_vdif,cross_corr,sdbe_preprocess
import h5py

import ctypes

VDIF_BYTE_SIZE = 1056
VDIF_BYTE_SIZE_DATA = 1024
VDIF_INT_SIZE = VDIF_BYTE_SIZE/4
VDIF_PER_BENG = 2048

BENG_CHANNELS_ = 16384
BENG_CHANNELS = (BENG_CHANNELS_ + 1)
BENG_SNAPSHOTS = 128

SWARM_RATE = 2496e6
R2DBE_RATE = 4096e6

BENG_BUFFER_IN_COUNTS = 40
SNAPSHOTS_PER_BATCH = 39

class sdbe_cupreprocess(object): 

  def __init__(self,gpuid,resamp_kind='fft',debias=None):
    ''' 
    Prep GPU for preprocessing

    resamp_kind : Choice of fft, linear, nearest
    debias: provide arrays for debiasing SWARM spectra ([2,N])
    '''

    # basic info   
    self.logger = logging.getLogger(__name__)
    self.gpuid = gpuid
    self.num_beng_counts = BENG_BUFFER_IN_COUNTS-1
    self.num_swarm_samples = int((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*2*BENG_CHANNELS_)
    self.num_r2dbe_samples = int(self.num_swarm_samples*R2DBE_RATE/SWARM_RATE)
    self.__resamp_kind = resamp_kind
    self.debias = False if debias is None else True

    # compile CUDA kernels
    kernel_source = kernel_template % {'BENG_BUFFER_IN_COUNTS':BENG_BUFFER_IN_COUNTS}
    kernel_module = SourceModule(kernel_source)
    self.__vdif_to_beng = kernel_module.get_function('vdif_to_beng')
    self.__reorderTzp_smem = kernel_module.get_function('reorderTzp_smem')
    #self.__reorderTz_smem = kernel_module.get_function('reorderTz_smem')
    self.__quantize2bit = kernel_module.get_function('quantize2bit')
    self.__zero_rout = kernel_module.get_function('zero_rout')
    self.__detrend = kernel_module.get_function('detrend')

    if self.__resamp_kind == 'linear':
      self.__linear_interp = kernel_module.get_function('linear')
      # pre-compute weights and indices for interpolation
      template_size = int(39*2*BENG_CHANNELS_*R2DBE_RATE/SWARM_RATE)
      ctid = SWARM_RATE/R2DBE_RATE*arange(template_size)
      cpu_ida  = floor(SWARM_RATE/R2DBE_RATE*arange(template_size)).astype(int32)
      cpu_wgt  = (ctid-cpu_ida).astype(float32)
      # and load these to the GPU
      self.__gpu_ida = cuda.mem_alloc(4 * template_size)
      cuda.memcpy_htod(self.__gpu_ida,cpu_ida)
      self.__gpu_wgt = cuda.mem_alloc(4 * template_size)
      cuda.memcpy_htod(self.__gpu_wgt,cpu_wgt)
    elif self.__resamp_kind == 'nearest': 
      self.__nearest_interp = kernel_module.get_function('nearest')

    # prep for debias ('optional')
    if self.debias:
      self.__gpu_xs0avg = cuda.mem_alloc(4*debias.shape[-1])
      self.__gpu_xs1avg = cuda.mem_alloc(4*debias.shape[-1])
      cuda.memcpy_htod(self.__gpu_xs0avg,debias[0,:])
      cuda.memcpy_htod(self.__gpu_xs1avg,debias[1,:])

    # inverse in-place FFT plan
    n = array([2*BENG_CHANNELS_],int32)
    inembed = array([16400],int32)
    onembed = array([2*BENG_CHANNELS_],int32)
    self.__sdbe_batch = BENG_SNAPSHOTS
    self.__plan_A = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, 16400,
                                        onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
 				         cufft.CUFFT_C2R, self.__sdbe_batch)

    # initalize FFT plans:
    if self.__resamp_kind == 'linear' or self.__resamp_kind == 'nearest':

      # band trimming R2C FFT plan
      n = array([4096],int32)
      inembed = array([4096],int32)
      onembed = array([4096/2+1],int32)
      self.__bandlimit_batch = self.num_r2dbe_samples / 4096 / 128 # we are memory limited.
      self.__plan_B = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, 4096,
					       onembed.ctypes.data, 1, 4096/2+1,
					       cufft.CUFFT_R2C, self.__bandlimit_batch)
      # band trimming C2R FFT plan
      n = array([2048],int32)
      inembed = array([2048/2+1],int32)
      onembed = array([2048],int32)
      self.__plan_C = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, 4096/2+1,
					       onembed.ctypes.data, 1, 2048,
					       cufft.CUFFT_C2R, self.__bandlimit_batch)
    elif self.__resamp_kind == 'fft':
      # inverse FFT plan
      #n = array([2*BENG_CHANNELS_],int32)
      #inembed = array([BENG_CHANNELS],int32)
      #onembed = array([2*BENG_CHANNELS_],int32)
      #self.__plan_A = cufft.cufftPlanMany(1, n.ctypes.data, inembed.ctypes.data, 1, BENG_CHANNELS,
      #                                onembed.ctypes.data, 1, 2*BENG_CHANNELS_,
 	#			       cufft.CUFFT_C2R, self.num_beng_counts*BENG_SNAPSHOTS)

      self.__bandlimit_batch = 32
      # Turn concatenated SWARM time series into single spectrum
      n = array([39*2*BENG_CHANNELS_],int32)
      inembed = array([39*2*BENG_CHANNELS_],int32)
      onembed = array([39*BENG_CHANNELS_+1],int32)
      self.__plan_B = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,39*2*BENG_CHANNELS_,
					onembed.ctypes.data,1,39*BENG_CHANNELS_+1,
					cufft.CUFFT_R2C,self.__bandlimit_batch)

      # Turn trimmed spectrum into 2048 timeseries
      n = array([32*2*BENG_CHANNELS_],int32)
      inembed = array([39*BENG_CHANNELS_+1],int32)
      onembed = array([32*2*BENG_CHANNELS_],int32)
      self.__plan_C = cufft.cufftPlanMany(1,n.ctypes.data,
					inembed.ctypes.data,1,39*BENG_CHANNELS_+1,
					onembed.ctypes.data,1,32*2*BENG_CHANNELS_,
					cufft.CUFFT_C2R,self.__bandlimit_batch)


    # deal with pagelocked memory for faster memory transfers.
    self.vdif_buf = cuda.pagelocked_empty((VDIF_INT_SIZE*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS,),uint32)
    self.quantized_0 = cuda.pagelocked_empty((self.num_r2dbe_samples/2/16,),uint32)
    self.quantized_1 = cuda.pagelocked_empty((self.num_r2dbe_samples/2/16,),uint32)

    self.__gpu_vdif_buf = None
    self.__gpu_beng_data_0 = None
    self.__gpu_beng_data_1 = None
    self.__gpu_beng_0 = None
    self.__gpu_beng_1 = None
    self.__gpu_time_series_0 = None
    self.__gpu_time_series_1 = None
    self.__gpu_quantized_0 = None
    self.__gpu_quantized_1 = None


  def memcpy_sdbe_vdif(self,bcount_offset):
    ''' Move sdbe vdif buffer buffer to device.'''
    self.bcount_offset = bcount_offset
    self.__gpu_vdif_buf = cuda.mem_alloc(VDIF_BYTE_SIZE*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS)
    cuda.memcpy_htod(self.__gpu_vdif_buf,self.vdif_buf)

  def memcpy_streams(self):
    ''' move quantized streams to host '''
    cuda.memcpy_dtoh(self.quantized_0,self.__gpu_quantized_0)
    cuda.memcpy_dtoh(self.quantized_1,self.__gpu_quantized_1)
    self.__gpu_quantized_0.free()
    self.__gpu_quantized_1.free()

  def depacketize_sdbe_vdif(self,blocks_per_grid=128):
    ''' 
    depacketize sdbe vdif:
    consecutive snapshots are indexed fastest 
    '''
    self.__gpu_beng_data_0 = cuda.mem_alloc(8*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS) # 15.6%
    self.__gpu_beng_data_1 = cuda.mem_alloc(8*BENG_CHANNELS_*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS) # 15.6%
    gpu_fid         = cuda.mem_alloc(4*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS)
    gpu_cid         = cuda.mem_alloc(4*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS)
    gpu_bcount      = cuda.mem_alloc(4*VDIF_PER_BENG*BENG_BUFFER_IN_COUNTS)
    gpu_beng_frame_completion = cuda.mem_alloc(4*BENG_BUFFER_IN_COUNTS)
    # launch kernel
    self.__vdif_to_beng(self.__gpu_vdif_buf,
			gpu_fid, 
			gpu_cid, 
			gpu_bcount,
			self.__gpu_beng_data_0,
			self.__gpu_beng_data_1,
			gpu_beng_frame_completion,
			int32(BENG_BUFFER_IN_COUNTS*VDIF_PER_BENG),
			int32(self.bcount_offset),
  			block=(32,32,1), grid=(blocks_per_grid,1,1))
    # we might need these later, but let's free the memory for now.
    gpu_fid.free()
    gpu_cid.free()
    gpu_bcount.free()
    gpu_beng_frame_completion.free()
    # could free 2-bit vdif (2%)
    self.__gpu_vdif_buf.free()

  def reorder_beng_data(self):
    '''
    reorder B-engine data with a shift by 2 snapshots where (channel index / 4) is even
    and then by 1 B-frame where snapshot index > 68.
    This function also debiases the spectra if template was initialized.
    '''
    self.__gpu_beng_0 = cuda.mem_alloc(8*16400*BENG_SNAPSHOTS*(BENG_BUFFER_IN_COUNTS-1))
    self.__gpu_beng_1 = cuda.mem_alloc(8*16400*BENG_SNAPSHOTS*(BENG_BUFFER_IN_COUNTS-1))
    self.__reorderTzp_smem(self.__gpu_beng_data_0,self.__gpu_beng_0,int32(BENG_BUFFER_IN_COUNTS),
		block=(16,16,1),grid=(BENG_CHANNELS_*BENG_SNAPSHOTS/(16*16),1,1),)
    self.__reorderTzp_smem(self.__gpu_beng_data_1,self.__gpu_beng_1,int32(BENG_BUFFER_IN_COUNTS),
		block=(16,16,1),grid=(BENG_CHANNELS_*BENG_SNAPSHOTS/(16*16),1,1),)
    # free unpacked, unordered b-frames
    self.__gpu_beng_data_0.free()
    self.__gpu_beng_data_1.free()

    if self.debias:
      self.__debias()

  def __debias(self):
    block = (128,1,1)
    grid = (BENG_CHANNELS_/128,1) 
    self.__detrend(self.__gpu_beng_0, int32(self.num_beng_counts*BENG_SNAPSHOTS),self.__gpu_xs0avg,
		block=block,grid=grid)
    self.__detrend(self.__gpu_beng_1, int32(self.num_beng_counts*BENG_SNAPSHOTS),self.__gpu_xs1avg,
		block=block,grid=grid)
   #__global__ void detrend(cufftComplex *spectra, int32_t N, float *avg){

  def resamp(self):
    if self.__resamp_kind == 'linear':
      self.__fft_linear_interp()
    elif self.__resamp_kind == 'fft':
      self.__fft_resample()
    elif self.__resamp_kind == 'nearest':
      self.__fft_nearest_interp()
    else:
      print 'no resampling kernel to match'


  def __fft_resample(self):
    '''
    Resample using FFTs.
    individual phased sums resampled at 2048 MHz. 
    '''

    #device memory allocation
    self.__gpu_time_series_0 = cuda.mem_alloc(4 * self.num_r2dbe_samples / 2) # 2048 MHz clock
    self.__gpu_time_series_1 = cuda.mem_alloc(4 * self.num_r2dbe_samples / 2) # 2048 MHz clock
    gpu_swarm = cuda.mem_alloc(4 * self.num_swarm_samples)

    # loop over phased sums
    for (phased_sum_in,phased_sum_out) in zip((self.__gpu_beng_0, self.__gpu_beng_1),(self.__gpu_time_series_0,self.__gpu_time_series_1)):
      # turn SWARM snapshots into timeseries
      for ib in range(self.num_beng_counts):
        cufft.cufftExecC2R(self.__plan_A,int(phased_sum_in)+int(8*ib*BENG_SNAPSHOTS*16400),int(gpu_swarm)+int(4*ib*BENG_SNAPSHOTS*2*BENG_CHANNELS_))
      phased_sum_in.free()
 
      gpu_tmp = cuda.mem_alloc(8*int(39*BENG_CHANNELS_+1)*self.__bandlimit_batch)
      for ib in range((BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS/39/self.__bandlimit_batch):
        # Turn concatenated SWARM time series into single spectrum
        cufft.cufftExecR2C(self.__plan_B,int(gpu_swarm)+
						int(4*39*2*BENG_CHANNELS_*self.__bandlimit_batch*ib),int(gpu_tmp))
        # Turn padded SWARM spectrum into time series with R2DBE sampling rate
        cufft.cufftExecC2R(self.__plan_C,
			int(gpu_tmp)+int(8*150*512),int(phased_sum_out)+
			int(4*32*2*BENG_CHANNELS_*ib*self.__bandlimit_batch))
      gpu_tmp.free()

    gpu_swarm.free()

  def __fft_nearest_interp(self):
    '''
    Resample using nearest interpolation.
    '''
    threads_per_block = 512
    blocks_per_grid = int(ceil(1. * self.num_r2dbe_samples / threads_per_block))

    # allocate device memory
    self.__gpu_time_series_1 = cuda.mem_alloc(4 * self.num_r2dbe_samples / 2) # 2048 MHz clock
    self.__gpu_time_series_0 = cuda.mem_alloc(4 * self.num_r2dbe_samples / 2) # 2048 MHz clock
    gpu_r2dbe_spec = cuda.mem_alloc(8 * (4096/2+1) * self.__bandlimit_batch) # memory peanuts
    gpu_swarm = cuda.mem_alloc(4 * self.num_swarm_samples)

    for (phased_sum_in,phased_sum_out) in zip((self.__gpu_beng_0, self.__gpu_beng_1),(self.__gpu_time_series_0,self.__gpu_time_series_1)):
      # Turn SWARM snapshots into timeseries
      for ib in range(self.num_beng_counts):
        cufft.cufftExecC2R(self.__plan_A,int(phased_sum_in)+int(8*ib*BENG_SNAPSHOTS*16400),int(gpu_swarm)+int(4*ib*BENG_SNAPSHOTS*2*BENG_CHANNELS_))
      phased_sum_in.free()
      # resample 
      gpu_resamp = cuda.mem_alloc(4 * self.num_r2dbe_samples) # 25% of device memory
      self.__nearest_interp(gpu_swarm,
		int32(self.num_swarm_samples),
		gpu_resamp,
		int32(self.num_r2dbe_samples),
		float64(SWARM_RATE/R2DBE_RATE),
		block=(threads_per_block,1,1),grid=(blocks_per_grid,1))

      # loop through resampled time series in chunks of batch_B num_r2dbe_samples/4096/__bandlimit_batch
      for ib in range(self.num_r2dbe_samples/4096/self.__bandlimit_batch):
        # compute spectrum with 4096 MHz sample clock
        cufft.cufftExecR2C(self.__plan_B,
        int(gpu_resamp)+int(4*ib*4096*self.__bandlimit_batch),int(gpu_r2dbe_spec))
        # invert to time series with BW of 1024 MHz, masking out first 150 MHz and last (1024-150) MHz. (pointers are 4-byte values)
        cufft.cufftExecC2R(self.__plan_C,
		int(gpu_r2dbe_spec)+int(8*150),
		int(phased_sum_out) + 
		int(4*ib*2048*self.__bandlimit_batch))

      gpu_resamp.free()

    gpu_r2dbe_spec.free()
    gpu_swarm.free()

  def __fft_linear_interp(self):
    '''
    Resample using linear interpolation.
    '''
    threads_per_block = 512
    blocks_per_grid = int(ceil(1. * self.num_r2dbe_samples / threads_per_block))

    # allocate device memory
    self.__gpu_time_series_1 = cuda.mem_alloc(4 * self.num_r2dbe_samples / 2) # 2048 MHz clock, 12.5% memory
    self.__gpu_time_series_0 = cuda.mem_alloc(4 * self.num_r2dbe_samples / 2) # 2048 MHz clock, 12.5% memory
    gpu_r2dbe_spec = cuda.mem_alloc(8 * (4096/2+1) * self.__bandlimit_batch) # memory peanuts
    gpu_swarm = cuda.mem_alloc(4 * self.num_swarm_samples)

    for (phased_sum_in,phased_sum_out) in zip((self.__gpu_beng_0, self.__gpu_beng_1),(self.__gpu_time_series_0,self.__gpu_time_series_1)):
      # Turn SWARM snapshots into timeseries
      for ib in range(self.num_beng_counts):
        cufft.cufftExecC2R(self.__plan_A,int(phased_sum_in)+int(8*ib*BENG_SNAPSHOTS*16400),int(gpu_swarm)+int(4*ib*BENG_SNAPSHOTS*2*BENG_CHANNELS_))
      phased_sum_in.free()
      ## resample 
      gpu_resamp = cuda.mem_alloc(4 * self.num_r2dbe_samples) # 25% of device memory
      self.__linear_interp(gpu_swarm,gpu_resamp,int32(self.num_swarm_samples),self.__gpu_wgt,self.__gpu_ida,
		block=(512,1,1),grid=(2097152/512,1))

      # loop through resampled time series in chunks of batch_B num_r2dbe_samples/4096/__bandlimit_batch
      for ib in range(self.num_r2dbe_samples/4096/self.__bandlimit_batch):
        # compute spectrum with 4096 MHz sample clock
        cufft.cufftExecR2C(self.__plan_B,int(gpu_resamp)+int(4*ib*4096*self.__bandlimit_batch),int(gpu_r2dbe_spec))
        # invert to time series with BW of 1024 MHz, masking out first 150 MHz and last (1024-150) MHz. (pointers are 4-byte values)
        cufft.cufftExecC2R(self.__plan_C,
		int(gpu_r2dbe_spec)+int(8*150),
		int(phased_sum_out) + 
		int(4*ib*2048*self.__bandlimit_batch))

      gpu_resamp.free()

    gpu_r2dbe_spec.free()
    gpu_swarm.free()

  def quantize(self,threshold):
    ''' quantize to 2-bits. note little endian format following vdif '''
    # zero out arrays
    num_int_words = self.num_r2dbe_samples / 2 / 16
    self.__gpu_quantized_0 = cuda.mem_alloc(2 * (self.num_r2dbe_samples / 2) / 8 )
    self.__gpu_quantized_1 = cuda.mem_alloc(2 * (self.num_r2dbe_samples / 2) / 8)
    self.__zero_rout(self.__gpu_quantized_0, int32(num_int_words),
		block=(512,1,1),grid=(num_int_words / 512,1))
    self.__zero_rout(self.__gpu_quantized_1, int32(num_int_words),
		block=(512,1,1),grid=(num_int_words / 512,1))
    # quantize
    threads_per_block = (16,1,1)
    blocks_per_grid=(64,1,1)
    self.__quantize2bit(self.__gpu_time_series_0,self.__gpu_quantized_0,int32(self.num_r2dbe_samples/2),float32(threshold[0]),
		block=threads_per_block,grid=blocks_per_grid)
    self.__quantize2bit(self.__gpu_time_series_1,self.__gpu_quantized_1,int32(self.num_r2dbe_samples/2),float32(threshold[1]),
		block=threads_per_block,grid=blocks_per_grid)

    # can free gpu_time_series_{0-1} now
    #self.__gpu_time_series_1.free()
    #self.__gpu_time_series_0.free()

  def gpumeminfo(self,driver,msg=''):
    self.logger.info('Memory usage: %f :: (%20s)' % (1.- 1.*driver.mem_get_info()[0]/driver.mem_get_info()[1],msg))

  def cleanup(self):

    # destroy plans
    cufft.cufftDestroy(self.__plan_A)
    cufft.cufftDestroy(self.__plan_B)
    cufft.cufftDestroy(self.__plan_C)

#################################################################

def get_bcount_from_vdif(vdif_start):
  '''
  read .vdif file into uint32 word array 
  '''
  BENG_VDIF_HDR_0_OFFSET_INT = 4  # b1 b2 b3 b4
  BENG_VDIF_HDR_1_OFFSET_INT = 5  #  c  0  f b0
  return ((vdif_start[BENG_VDIF_HDR_1_OFFSET_INT]&0xFF000000)>>24) + ((vdif_start[BENG_VDIF_HDR_0_OFFSET_INT]&0x00FFFFFF)<<8);

def read_sdbe_vdif(vdif_buf,filename_input,num_vdif_frames,beng_frame_offset=1,batched=True):
  logger = logging.getLogger(__name__)
  #vdif_buf = empty(VDIF_BYTE_SIZE*num_vdif_frames/4, dtype=uint32)
  for ii in arange(4):
    filename_data = "%s_swarmdbe_eth%d.vdif" % (filename_input, 2+ii)
    fh = open(filename_data,"r")
    # seek to specified B-engine frame boundary
    tmp_vdif = array(unpack('<{0}I'.format(VDIF_INT_SIZE),fh.read(VDIF_BYTE_SIZE)), uint32)
    tmp_bcount_curr = get_bcount_from_vdif(tmp_vdif)
    tmp_bcount_prev = tmp_bcount_curr
    while (tmp_bcount_curr-tmp_bcount_prev < beng_frame_offset):
      tmp_vdif = array(unpack('<{0}I'.format(VDIF_INT_SIZE),fh.read(VDIF_BYTE_SIZE)), uint32)
      tmp_bcount_curr = get_bcount_from_vdif(tmp_vdif)
    logger.debug('B-count = %d = %d, seeking one frame back' % (tmp_bcount_curr, tmp_bcount_prev+beng_frame_offset))
    fh.seek(-1*VDIF_BYTE_SIZE,1)
    bcount_offset = tmp_bcount_curr
    # read data
    vdif_buf[ii*VDIF_INT_SIZE*num_vdif_frames/4:(ii+1)*VDIF_INT_SIZE*num_vdif_frames/4] = array(unpack('<{0}I'.format(VDIF_INT_SIZE*num_vdif_frames/4),fh.read(VDIF_BYTE_SIZE * num_vdif_frames / 4)), uint32)
    fh.seek(-1*VDIF_BYTE_SIZE,1)
    tmp_vdif = array(unpack('<{0}I'.format(VDIF_INT_SIZE),fh.read(VDIF_BYTE_SIZE)), uint32)
    logger.debug('ending with B-count = %d, read %d B-engine frames' % (get_bcount_from_vdif(tmp_vdif), get_bcount_from_vdif(tmp_vdif)-bcount_offset+1))
    fh.close()
  return bcount_offset

#################################################################

if __name__ == "__main__":
  import pycuda.autoinit
  #cuda.init()

  parser = ArgumentParser(description="Convert SWARM vdif data into 4096 Msps time-domain vdif data")
  parser.add_argument('-v', dest='verbose', action='store_true', help='display debug information')
  parser.add_argument('-b', dest='basename', help='scan filename base for SWARM data', default='prep6_test1_local')
  parser.add_argument('-d', dest='dir', help='directory for input SWARM data products', default='/home/shared/sdbe_preprocessed/')
  parser.add_argument('-s', dest='skip', type=int, help='starting snapshot', default=1)
  parser.add_argument('-alg', dest='resamp_kind', type=str, help='resampling algorithm', default='fft')
  parser.add_argument('-c', dest='correlate', action='store_true', help='correlate against R2DBE')
  parser.add_argument('-debias', dest='debias', action='store_true', help='debias SWARM spectra')
  args = parser.parse_args()

  logger = logging.getLogger(__name__)
  logger.addHandler(logging.StreamHandler())
  logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

  # use precomputed thresholds for quantization
  if args.resamp_kind == 'linear':
    thresh = [741508.44,773105.12]
  elif args.resamp_kind == 'fft':
    thresh = [2.80364832e+08, 2.95670688e+08]
  elif args.resamp_kind == 'nearest':
    thresh = [833892.25, 875652.25]
  thresh = sqrt(2)*erfinv(0.5)*array(thresh)

  if args.debias:
    # read in debiasing template 
    f = open(args.basename+'_spline.bin','rb')
    debias = array(unpack('%df' % (2*BENG_CHANNELS_,),f.read(4*2*BENG_CHANNELS_)),dtype=float32).reshape(2,BENG_CHANNELS_)
    f.close()
    # zero out SWARM spurs appearing every 2048 channels
    spurs = arange(2048, BENG_CHANNELS_, 2048)
    debias[:,spurs] = 0
    # remove guard band artifacts
    debias[0,:20] = 0 
    debias[1,15200:] = 0 
    # pad to match 16400 padded spectra chunks
    debias = hstack([debias, zeros((2,16400 - BENG_CHANNELS_), dtype=float32) ]) 

  # timers
  clock = defaultdict(float)
  counts = defaultdict(int)
  total = defaultdict(float)
  def tic(name):
    clock[name] = timer()
  def toc(name):
    counts[name] = counts[name] + 1
    total[name] = total[name] + timer() - clock[name]

  if True: 

    g = sdbe_cupreprocess(0,resamp_kind=args.resamp_kind,debias=debias)

    # load vdif onto device
    bcount_offset = read_sdbe_vdif(g.vdif_buf,args.dir+args.basename,BENG_BUFFER_IN_COUNTS*VDIF_PER_BENG,
						beng_frame_offset=g.gpuid*BENG_BUFFER_IN_COUNTS+1)

    tic('gpu total')

    tic('mem transfer')
    g.memcpy_sdbe_vdif(bcount_offset)
    toc('mem transfer')
  
    # depacketize vdif
    tic('depacketize')
    g.depacketize_sdbe_vdif()
    toc('depacketize')
  
    # reorder B-engine data
    tic('reorder  ')
    g.reorder_beng_data()
    toc('reorder  ')
  
    # linear resampling
    tic('resample')
    g.resamp()
    toc('resample')

    # optional: calculate threshold for quantization
  
    #quantize
    tic('quantize')
    g.quantize(thresh)
    toc('quantize')

    # load quantized data
    tic('mem transfer')
    g.memcpy_streams()
    toc('mem transfer')
  
    toc('gpu total')

    cpu_q_r2dbe = empty(g.num_r2dbe_samples/2,dtype=int32)
    for i in range(16):
      cpu_q_r2dbe[i::16] = (g.quantized_1 & (0x3 << 2*i)) >> (2*i)

  # clean up
  #for g in gpus:
    #g.cleanup()
  
  # report timing info
  real_time = (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*2*BENG_CHANNELS_/SWARM_RATE
  print "%s\t%s\t%s\t%s" % ('operation', 'counts', 'total [ms]', 'x(real = %.3f)' % (real_time*1e3))
  print "-----------------------------------------------------------"
  for k in total.keys():
      print "%s\t%d\t%.2f\t\t%.3f" % (k, counts[k], 1e3*total[k], total[k]/real_time )

  if args.correlate:
    import matplotlib.pyplot as plt
    logger.info("\nINFO:: Correlating result again R2DBE")

    # pull float data from device
    cpu_r2dbe = empty(g.num_r2dbe_samples/2,dtype=float32) # empty array for result
    cuda.memcpy_dtoh(cpu_r2dbe,g._sdbe_cupreprocess__gpu_time_series_1)
  
    # Now read R2DBE data covering roughly the same time window as the SWARM
    # data. Start at an offset of zero (i.e. from the first VDIF packet) to
    # keep things simple.
    N_r_vdif_frames = int(ceil(BENG_SNAPSHOTS*(BENG_BUFFER_IN_COUNTS-1)*R2DBE_RATE/SWARM_RATE))
    vdif_frames_offset = 0
    d = sdbe_preprocess.get_diagnostics_from_file(args.basename,rel_path=args.dir)
    xr = read_r2dbe_vdif.read_from_file(args.dir+args.basename+'_r2dbe_eth3.vdif',
		N_r_vdif_frames,vdif_frames_offset)
  
    # compare to pre-shifted data
    hf5 = h5py.File(args.dir+args.basename+'_sdbe_preprocess.hdf5')
    spectra = hf5.get('Xs1').value
    xs_shifted = sdbe_preprocess.resample_sdbe_to_r2dbe_fft_interp(spectra,interp_kind="linear")
  
    # now we check the band limited series
    xr_bl = sdbe_preprocess.bandlimit_1248_to_1024(xr[:(xr.size//4096)*4096],sub_sample=True)
    xs_bl = sdbe_preprocess.bandlimit_1248_to_1024(xs_shifted[:(xs_shifted.size//4096)*4096],sub_sample=True)
  
    # check correlation given pre-determined time shift
    s_range = arange(-16,16)
    s_avg = 128
    offset_swarmdbe_data = d['offset_swarmdbe_data']
    idx_offset = d['get_idx_offset'][0]
    fft_window_size = 32768

    #for (x,shift) in zip((cpu_q_r2dbe,xs_bl),(floor(idx_offset*2*BENG_CHANNELS_/SWARM_RATE*2048e6),0)):
    #for (x,shift) in zip((cpu_q_r2dbe,cpu_r2dbe,xs_bl),(floor(idx_offset*2*BENG_CHANNELS_/SWARM_RATE*2048e6),floor(idx_offset*2*BENG_CHANNELS_/SWARM_RATE*2048e6),0)):
    for (x,shift) in zip((cpu_q_r2dbe,cpu_r2dbe),2*(floor(idx_offset*2*BENG_CHANNELS_/SWARM_RATE*2048e6),)):
    #for (x,shift) in zip((cpu_q_r2dbe,),1*(floor(idx_offset*2*BENG_CHANNELS_/SWARM_RATE*2048e6),)):
  
      # FFT resampled
      x0 = xr_bl.copy()
      x1 = x[shift+offset_swarmdbe_data/2-1:]
      n = min((2**int(floor(log2(x0.size))),2**int(floor(log2(x1.size)))))
      x0 = x0[:n]
      x1 = x1[:n]
  
      X0 = rfft(x0.reshape(n/fft_window_size,fft_window_size),axis=-1)
      X1 = rfft(x1.reshape(n/fft_window_size,fft_window_size),axis=-1)
      X0*= exp(-1.0j*angle(X0[0,0]))
      X1*= exp(-1.0j*angle(X1[0,0]))
  
      S_0x0 = (X0 * X0.conj()).mean(axis=0)
      S_1x1 = (X1 * X1.conj()).mean(axis=0)
      S_0x1 = (X0 * X1.conj()).mean(axis=0)
      p = 8; q = 16
      s_0x0 = irfft(S_0x0,n=p*fft_window_size).real
      s_1x1 = irfft(S_1x1,n=p*fft_window_size).real
      s_0x1 = irfft(S_0x1,n=p*fft_window_size).real/sqrt(s_0x0.max()*s_1x1.max())
  
      print ' (max,min) corr coeffs:\t\t', s_0x1.max(), s_0x1.min()
      #plt.figure()
      #plt.stem(arange(-p*q/2,p*q/2),roll(s_0x1,p*q/2)[:p*q])
  
    plt.show()

#[krosenfe@hamster cuda]$ python sdbe_cupreprocessed.py -alg linear -c
#operation       counts  total [ms]      x(real = 65.536)
#-----------------------------------------------------------
#gpu total       1       331.82          5.063
#reorder         1       26.45           0.404
#resample        1       175.14          2.672
#depacketize     1       12.02           0.183
#quantize        1       0.32            0.005
#mem transfer    2       117.88          1.799
#
#INFO:: Correlating result again R2DBE
# (max,min) corr coeffs:         0.188520458271 -0.039532406453
# (max,min) corr coeffs:         0.315810267681 -0.0663339328473
#
#[krosenfe@hamster cuda]$ python sdbe_cupreprocessed.py -alg nearest -c
#operation       counts  total [ms]      x(real = 65.536)
#-----------------------------------------------------------
#gpu total       1       347.40          5.301
#reorder         1       26.46           0.404
#resample        1       177.43          2.707
#depacketize     1       12.04           0.184
#quantize        1       0.35            0.005
#mem transfer    2       131.11          2.001
#
#INFO:: Correlating result again R2DBE
# (max,min) corr coeffs:         0.185023550395 -0.0460234654388
# (max,min) corr coeffs:         0.309522793633 -0.0781857316486
#
#[krosenfe@hamster cuda]$ python sdbe_cupreprocessed.py -alg fft -c
#operation       counts  total [ms]      x(real = 65.536)
#-----------------------------------------------------------
#gpu total       1       372.88          5.690
#reorder         1       26.38           0.402
#resample        1       205.21          3.131
#depacketize     1       12.00           0.183
#quantize        1       0.33            0.005
#mem transfer    2       128.95          1.968
#
#INFO:: Correlating result again R2DBE
# (max,min) corr coeffs:         0.191704138582 -0.0565066308196
# (max,min) corr coeffs:         0.320292665479 -0.0944922613483
