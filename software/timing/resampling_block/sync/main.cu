/*
This program measures the timing performance of the 
APHIDS resampling block.  The speed of each individual kernel
is measured (not including memcpy costs).

@author Katherine Rosenfeld
@date 7/28/2015
*/

#include <stdio.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "vdif_inout_gpu_thread.h"

#define ROUNDS 10
#define QUANTIZE_THRESHOLD 1.f

void reportDeviceMemInfo(void){
#ifdef GPU_DEBUG
  size_t avail,total;
  cudaMemGetInfo(&avail,&total);
  fprintf(stdout, "GPU_DEBUG : Dev mem (avail, tot, frac): %u , %u  = %0.4f\n", 
		(unsigned)avail, (unsigned)total, 100.f*avail/total);
#endif
}

static void HandleError( cudaError_t err,const char *file,int line ) {
#ifdef GPU_DEBUG
    if (err != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
    }
#endif
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef struct aphids_resampler {
    int deviceId;
    int skip_chan;
    cufftComplex *gpu_A_0, *gpu_A_1;
    cufftComplex *gpu_B_0, *gpu_B_1;
    int fft_size[3],batch[3],repeat[3];
    cufftHandle cufft_plan[3];
    cudaStream_t stream;
    cudaEvent_t tic,toc;

} aphids_resampler_t;

/** @brief Initialize resampler structure (including device memory).
 */
int aphids_resampler_init(aphids_resampler_t *resampler, int _deviceId) {

  // create FFT plans
  int inembed[3], onembed[3];

  resampler->deviceId = _deviceId;

  // switch to device
  cudaSetDevice(resampler->deviceId);

  size_t workSize[1];
  cudaStreamCreate(&(resampler->stream));
  cudaEventCreate(&(resampler->tic));
  cudaEventCreate(&(resampler->toc));

  // allocate device memory
  cudaMalloc((void **)&(resampler->gpu_A_0), BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS*BENG_CHANNELS_*sizeof(cufftComplex));	// 671088640B
  cudaMalloc((void **)&(resampler->gpu_A_1), BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS*BENG_CHANNELS_*sizeof(cufftComplex));	// 671088640B
  cudaMalloc((void **)&(resampler->gpu_B_0), (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS*sizeof(cufftComplex));	// 654950400B
  cudaMalloc((void **)&(resampler->gpu_B_1), (BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS*sizeof(cufftComplex));	// 654950400B

 /*
 * http://docs.nvidia.com/cuda/cufft/index.html#cufft-setup
 * iFFT transforming complex SWARM spectra into real time series.
 * Input data is padded according to UNPACKED_BENG_CHANNELS >= BENG_CHANNELS.
 * Single batch size is BENG_SNAPHOTS and so to cover the entire databuf
 * this must repeated BENG_BUFFER_IN_COUNTS-1 times (saves 14.8% memory)
 */
  resampler->fft_size[0] = 2*BENG_CHANNELS_;
  inembed[0]  = UNPACKED_BENG_CHANNELS; 
  onembed[0]  = 2*BENG_CHANNELS_;
  resampler->batch[0]    = BENG_SNAPSHOTS;
  resampler->repeat[0]   = BENG_BUFFER_IN_COUNTS - 1;
  cufftPlanMany(&(resampler->cufft_plan[0]), 1, &(resampler->fft_size[0]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_C2R,resampler->batch[0]);
  cufftGetSize(resampler->cufft_plan[0],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 0 is %dx%dx%d\n", resampler->repeat[0],resampler->batch[0],resampler->fft_size[0]);
  fprintf(stdout,"GPU_DEBUG : plan 0 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();

  /*
 * FFT transforming time series into complex spectrum.
 * Input data has dimension RESAMPLING_CHUNK_SIZE with
 * Set the batch size to be RESAMPLING_BATCH with
 * (BENG_BUFFER_IN_COUNTS-1)*2*BENG_CHANNELS_*BENG_SNAPSHOTS/RESAMPLING_CHUNK_SIZE / RESAMPLING_BATCH
 *  required iterations.
 */
  resampler->fft_size[1] = RESAMPLING_CHUNK_SIZE;
  inembed[0]  = RESAMPLING_CHUNK_SIZE; 
  onembed[0]  = RESAMPLING_CHUNK_SIZE/2+1;
  resampler->batch[1]  = RESAMPLING_BATCH;
  resampler->repeat[1] = (BENG_BUFFER_IN_COUNTS-1)*2*BENG_CHANNELS_*BENG_SNAPSHOTS/RESAMPLING_CHUNK_SIZE/RESAMPLING_BATCH;
  cufftPlanMany(&(resampler->cufft_plan[1]), 1, &(resampler->fft_size[1]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_R2C,resampler->batch[1]);
  cufftGetSize(resampler->cufft_plan[1],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 1 is %dx%dx%d\n", resampler->repeat[1],resampler->batch[1],resampler->fft_size[1]);
  fprintf(stdout,"GPU_DEBUG : plan 1 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();

 /*
 * FFT transforming complex spectrum into resampled time series simultaneously
 * trimming the SWARM guard bands (150 MHz -- 1174 MHz).
 * Input data has dimension EXPANSIONS_FACTOR
 */
  resampler->skip_chan = (int)(150e6 / (SWARM_RATE / RESAMPLING_CHUNK_SIZE));
  resampler->fft_size[2] = RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR;
  inembed[0]  = RESAMPLING_CHUNK_SIZE/2+1; 
  onembed[0]  = RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR;
  resampler->batch[2]    = resampler->batch[1];
  resampler->repeat[2]   = resampler->repeat[1];
  cufftPlanMany(&(resampler->cufft_plan[2]), 1, &(resampler->fft_size[2]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_C2R,resampler->batch[2]);


#ifdef GPU_DEBUG
  for (i=0; i<3; ++i){
    cufftSetStream(resampler->cufft_plan[i], resampler->stream);
  }
  cufftGetSize(resampler->cufft_plan[2],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 2 is %dx%dx%d\n", resampler->repeat[2],resampler->batch[2],resampler->fft_size[2]);
  fprintf(stdout,"GPU_DEBUG : masking first %d channels \n",resampler->skip_chan);
  fprintf(stdout,"GPU_DEBUG : plan 2 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  

  return 1;
}

/**@brief Reorder B-engine data for BENG_FRAMES_OUT_CONSECUTIVE_SNAPSHOTS
Reorder B-engine data that is snapshot contiguous and where consecutive channels
are separated by 128 snapshots time the number of B-engine frames in the buffer.
This kernel must be called with a (16,16) thread block and enough x-blocks
to cover 1 B-engine.  The output has 16400 zero-padded channels
- blockDim.x = 16
- blockDim.y = 16;
- gridDim.x = BENG_CHANNELS_ * BENG_SNAPSHOTS / (blockDim.x*blockDim.y)
@author Katherine Rosenfeld
@date June 2015
*/
__global__ void reorderTzp_smem(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){
  // gridDim.x = 16384 * 128 / (16 * 16) = 8192
  // blockDim.x = 16; blockDim.y = 16;
  // --> launches 2097152 threads

  int32_t sid_out,bid_in;

  __shared__ cufftComplex tile[16][16];

  // for now, let us loop the grid over B-engine frames:
  for (int bid_out=0; bid_out<num_beng_frames-1; bid_out+=1){

    // input snapshot id
    int sid_in = (blockIdx.x * blockDim.x + threadIdx.x) & (BENG_SNAPSHOTS-1);
    // input channel id 
    int cid = threadIdx.y + blockDim.y * (blockIdx.x / (128 / blockDim.x));

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    // and by 1-B-frame:
    if (sid_out < 69){
      bid_in = bid_out;
    } else {
      bid_in = bid_out+1;
    }

    tile[threadIdx.x][threadIdx.y] = beng_data_in[BENG_SNAPSHOTS*num_beng_frames*cid + BENG_SNAPSHOTS*bid_in + sid_in];

    __syncthreads();

    // now we transpose warp orientation over channels and snapshot index

    // snapshot id
    sid_in = threadIdx.y + (blockIdx.x*blockDim.y) & (BENG_SNAPSHOTS-1);
    // channel id 
    cid = threadIdx.x + blockDim.x * (blockIdx.x / (BENG_SNAPSHOTS / blockDim.x)); 

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    beng_data_out[(BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS)*bid_out + UNPACKED_BENG_CHANNELS*sid_out + cid] 
	= tile[threadIdx.y][threadIdx.x];

    // zero out nyquist: 
    if (cid < 16) {
      beng_data_out[(BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS)*bid_out + UNPACKED_BENG_CHANNELS*sid_out + BENG_CHANNELS_]	= make_cuComplex(0.,0.);
    }

    __syncthreads();
  }
}

/** @brief 2-bit quantization kernel

This 2bit quantization kernel must be called with 16 x-threads,
any number of y-threads, and any number of x-blocks.
@author Andre Young
@date June 2015
*/
__global__ void quantize2bit(const float *in, unsigned int *out, int N, float thresh)
{
	int idx_in = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int idx_out = blockIdx.x*blockDim.y + threadIdx.y;
	
	for (int ii=0; (idx_in+ii)<N; ii+=gridDim.x*blockDim.x*blockDim.y)
	{
		//This is is for 00 = -2, 01 = -1, 10 = 0, 11 = 1.
		/* Assume sample x > 0, lower bit indicates above threshold. Then
		 * test, if x < 0, XOR with 11.
		 */
		//int sample_2bit = ( ((fabsf(in[idx_in+ii]) >= thresh) | 0x02) ^ (0x03*(in[idx_in+ii] < 0)) ) & 0x3;
		int sample_2bit = ( ((fabsf(in[idx_in+ii]) >= thresh) | 0x02) ^ (0x03*(in[idx_in+ii] < 0)) ) & OUTPUT_MAX_VALUE_MASK;
		//~ //This is for 11 = -2, 10 = -1, 01 = 0, 10 = 1
		//~ int sample_2bit = ((fabsf(in[idx_in+ii]) <= thresh) | ((in[idx_in+ii] < 0)<<1)) & OUTPUT_MAX_VALUE_MASK;
		sample_2bit = sample_2bit << (threadIdx.x*2);
		atomicOr(out+idx_out, sample_2bit);
		idx_out += gridDim.x*blockDim.y;
	}
}

/** @brief Transform SWARM spectra into timeseries
 */
int SwarmC2R(aphids_resampler_t *resampler){
  // transform SWARM spectra into timeseries
  for (int i = 0; i < resampler->repeat[0]; ++i) {
	cufftExecC2R(resampler->cufft_plan[0],
			resampler->gpu_B_0 + i*resampler->batch[0]*UNPACKED_BENG_CHANNELS, 
			(cufftReal *)resampler->gpu_A_0 + i*resampler->batch[0]*(2*BENG_CHANNELS_));
	}
  return 1;
}

/** @brief Transform SWARM timeseries into R2DBE compatible spectrum
 */
int SwarmR2C(aphids_resampler_t *resampler){
  // transform timeseries into reconfigured spectra
  for (int i = 0; i < resampler->repeat[1]; ++i) {
	cufftExecR2C(resampler->cufft_plan[1],
		(cufftReal *) resampler->gpu_A_0 + i*resampler->batch[1]*RESAMPLING_CHUNK_SIZE,
		resampler->gpu_B_0 + i*resampler->batch[1]*(RESAMPLING_CHUNK_SIZE/2+1));
  }
  return 1;
}

/** @brief Transform half R2DBE spectrum into timeseries
 */
int Hr2dbeC2R(aphids_resampler_t *resampler){
  for (int i = 0; i < resampler->repeat[2]; ++i) {
	cufftExecC2R(resampler->cufft_plan[2],
		resampler->gpu_B_0 + i*resampler->batch[2]*(RESAMPLING_CHUNK_SIZE/2+1) + resampler->skip_chan,
		(cufftReal *) resampler->gpu_A_0 + i*resampler->batch[2]*(RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR));
  }
  return 1;
}

// destructor
int aphids_resampler_destroy(aphids_resampler_t *resampler) {
  HANDLE_ERROR( cudaSetDevice(resampler->deviceId) );
  HANDLE_ERROR( cudaFree(resampler->gpu_A_0) );
  HANDLE_ERROR( cudaFree(resampler->gpu_B_0) );
  HANDLE_ERROR( cudaFree(resampler->gpu_A_1) );
  HANDLE_ERROR( cudaFree(resampler->gpu_B_1) );
  for (int i=0; i < 3; ++i){
	cufftDestroy(resampler->cufft_plan[i]);
  }
  cudaEventDestroy(resampler->tic);
  cudaEventDestroy(resampler->toc);
  HANDLE_ERROR( cudaStreamDestroy(resampler->stream) );
  HANDLE_ERROR( cudaDeviceReset() );
  return 1;
}

//*********************************************************

int main(int argc, char *argv[]){
  int numDevices;
  aphids_resampler_t resampler;
  float avgTime,elapsedTime;
  FILE *pFile;
  char buffer[50];


  // get number of devices
  cudaGetDeviceCount(&numDevices);

  // iterate of the number of devices
  for (int deviceId=0; deviceId < numDevices; deviceId++){
  	sprintf(buffer,"timing.%1d.dat",deviceId);
  	pFile = fopen(buffer,"w");

	// initialize resampler that device
	aphids_resampler_init(&resampler,deviceId);

	dim3 threads(16,16,1);
	dim3 blocks((BENG_CHANNELS_*BENG_SNAPSHOTS/(16*16)),1,1);
	avgTime = 0.;
	for (int cnt=0; cnt<ROUNDS; cnt++){
	   cudaEventRecord(resampler.tic);
	   reorderTzp_smem<<<blocks,threads>>>(resampler.gpu_A_0, resampler.gpu_B_0, BENG_BUFFER_IN_COUNTS);
	   cudaEventRecord(resampler.toc);
	   cudaEventSynchronize(resampler.toc);
	   cudaEventElapsedTime(&elapsedTime,resampler.tic,resampler.toc);
	   avgTime += elapsedTime;
	}
	avgTime /= ROUNDS;
	fprintf(pFile,"%15s %5i %12.4f %12.4f\n","reorderTzp_smem",deviceId, avgTime, 
		2*(BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*(2*BENG_CHANNELS_)/(elapsedTime * 1e-3) / 1e9);

  	// transform SWARM spectra to time series
	avgTime = 0.;
	for (int cnt=0; cnt<ROUNDS; cnt++){
	  cudaEventRecord(resampler.tic);
	  SwarmC2R(&resampler);
	  cudaEventRecord(resampler.toc);
	  cudaEventSynchronize(resampler.toc);
	  cudaEventElapsedTime(&elapsedTime,resampler.tic,resampler.toc);
	  avgTime += elapsedTime;
	}
	avgTime /= ROUNDS;
	fprintf(pFile,"%15s %5i %12.4f %12.4f\n","SwarmC2R",deviceId, avgTime, 
		2*(BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*(2*BENG_CHANNELS_)/(elapsedTime * 1e-3) / 1e9);

	// transform SWARM time series to R2DBE compatible spectra
	avgTime = 0.;
	for (int cnt=0; cnt<ROUNDS; cnt++){
	  cudaEventRecord(resampler.tic);
	  SwarmR2C(&resampler);
	  cudaEventRecord(resampler.toc);
	  cudaEventSynchronize(resampler.toc);
	  cudaEventElapsedTime(&elapsedTime,resampler.tic,resampler.toc);
	  avgTime += elapsedTime;
	}
	avgTime /= ROUNDS;
	fprintf(pFile,"%15s %5i %12.4f %12.4f\n","SwarmR2C",deviceId, avgTime, 
		2*(BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*(2*BENG_CHANNELS_)/(elapsedTime * 1e-3) / 1e9);

	// transform R2DBE spectra to trimmed and resampled time series
	avgTime = 0.;
	for (int cnt=0; cnt<ROUNDS; cnt++){
	  cudaEventRecord(resampler.tic);
	  Hr2dbeC2R(&resampler);
	  cudaEventRecord(resampler.toc);
	  cudaEventSynchronize(resampler.toc);
	  cudaEventElapsedTime(&elapsedTime,resampler.tic,resampler.toc);
	  avgTime += elapsedTime;
	}
	avgTime /= ROUNDS;
	fprintf(pFile,"%15s %5i %12.4f %12.4f\n","Hr2dbeC2R",deviceId, avgTime, 
		2*(BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*(2*BENG_CHANNELS_)/(elapsedTime * 1e-3) / 1e9);

 	// quantize to 2-bits
	threads.x = 16; threads.y = 32; threads.z = 1;
	blocks.x = 512; blocks.y = 1; blocks.z = 1;
	avgTime = 0.;
	for (int cnt=0; cnt<ROUNDS; cnt++){
	  cudaEventRecord(resampler.tic);
	  quantize2bit<<<blocks,threads>>>((float *) resampler.gpu_A_0, (unsigned int*) resampler.gpu_B_0, 
		(2*BENG_CHANNELS_*BENG_SNAPSHOTS*EXPANSION_FACTOR),
		QUANTIZE_THRESHOLD);
	  cudaEventRecord(resampler.toc);
	  cudaEventSynchronize(resampler.toc);
	  cudaEventElapsedTime(&elapsedTime,resampler.tic,resampler.toc);
	  avgTime += elapsedTime;
	}
	avgTime /= ROUNDS;
	fprintf(pFile,"%15s %5i %12.4f %12.4f\n","quantize2bit",deviceId, avgTime, 
		2*(BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS*(2*BENG_CHANNELS_)/(elapsedTime * 1e-3) / 1e9);

	// destroy resampler
	aphids_resampler_destroy(&resampler);
  	fclose(pFile);
  }

}
