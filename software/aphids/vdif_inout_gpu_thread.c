#include <stdio.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/time.h>

#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"
#include "vdif_out_databuf.h"
#include "vdif_upload_to_gpu_databuf.h"
#include "vdif_inout_gpu_thread.h"

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cufft.h>

#define GPU_DEBUG
#define GPU_COMPUTE

#define STATE_ERROR    -1
#define STATE_INIT      0
#define STATE_PROC      1

#ifdef GPU_DEBUG
void reportDeviceMemInfo(void){
  size_t avail,total;
  cudaMemGetInfo(&avail,&total);
  fprintf(stdout, "GPU_DEBUG : Dev mem (avail, tot, frac): %u , %u  = %0.4f\n", 
		(unsigned)avail, (unsigned)total, 100.f*avail/total);
}
#endif

/*
static void HandleError( cudaError_t err,const char *file,int line ) {
#ifdef GPU_DEBUG
    if (err != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
    }
#endif
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
*/

static void *run_method(hashpipe_thread_args_t * args) {

  int i = 0;
  int rv = 0;
  int index_in = 0;
  int index_out = 0;
  vdif_in_packet_block_t this_vdif_packet_block;
  vdif_in_databuf_t *db_in = (vdif_in_databuf_t *)args->ibuf;
  vdif_out_databuf_t *db_out = (vdif_out_databuf_t *)args->obuf;
  aphids_context_t aphids_ctx;
  int state = STATE_INIT;

  // CUDA variables
  int fft_size[1], inembed[1], onembed[1], skip_chan;
  cufftResult cufft_status = CUFFT_SUCCESS;
  int batch[3],repeat[3];
  cufftHandle cufft_plan[3];
  size_t workSize[1];
  cufftComplex *gpu_a;
  cufftReal *gpu_b;
  size_t avail,total;
  cudaStream_t stream;
#ifdef GPU_DEBUG
  cudaEvent_t tic,toc;
  float elapsedTime;
#endif

  // Initialize CUDA memory arrays
  cudaSetDevice(0);
  cudaStreamCreate( &stream );
  cudaMalloc((void **)&gpu_a, UNPACKED_BENG_CHANNELS*BENG_SNAPSHOTS*(BENG_FRAMES_PER_BLOCK-1)*sizeof(cufftComplex));
  cudaMalloc((void **)&gpu_b, BENG_SNAPSHOTS*(BENG_FRAMES_PER_BLOCK-1)*2*BENG_CHANNELS_*sizeof(cufftReal));
#ifdef GPU_DEBUG
  if (cudaGetLastError() != cudaSuccess){
    hashpipe_error(__FILE__, "Cuda error: failed to allocate memory");
    state = STATE_ERROR;
  }
  cudaEventCreate(&tic);
  cudaEventCreate(&toc);
#endif

  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
    return NULL;
  }

  /* GPU initialization code */
 /*
 * http://docs.nvidia.com/cuda/cufft/index.html#cufft-setup
 * iFFT transforming complex SWARM spectra into real time series.
 * Input data is padded according to UNPACKED_BENG_CHANNELS >= BENG_CHANNELS.
 * Single batch size is BENG_SNAPHOTS and so to cover the entire databuf
 * this must repeated BENG_FRAMES_PER_BLOCK-1 times (saves 14.8% memory)
 */
  fft_size[0] = 2*BENG_CHANNELS_;
  inembed[0]  = UNPACKED_BENG_CHANNELS; 
  onembed[0]  = 2*BENG_CHANNELS_;
  batch[0]    = BENG_SNAPSHOTS;
  repeat[0]   = BENG_FRAMES_PER_BLOCK - 1;
  //batch[0]    = BENG_SNAPSHOTS * (BENG_FRAMES_PER_BLOCK - 1);
  //repeat[0]   = 1; 
  cufft_status = cufftPlanMany(&(cufft_plan[0]), 1, &(fft_size[0]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_C2R,batch[0]);
  if (cufft_status != CUFFT_SUCCESS)
  {
    hashpipe_error(__FILE__, "CUFFT error: plan 0 creation failed");
    state = STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cufftGetSize(cufft_plan[0],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 0 is %dx%dx%d\n", repeat[0],batch[0],fft_size[0]);
  fprintf(stdout,"GPU_DEBUG : plan 0 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  

  /*
 * FFT transforming time series into complex spectrum.
 * Input data has dimension RESAMPLING_CHUNK_SIZE with
 * Set the batch size to be RESAMPLING_BATCH with
 * (BENG_FRAMES_PER_BLOCK-1)*2*BENG_CHANNELS_*BENG_SNAPSHOTS/RESAMPLING_CHUNK_SIZE / RESAMPLING_BATCH
 *  required iterations.
 */
  fft_size[1] = RESAMPLING_CHUNK_SIZE;
  inembed[0]  = RESAMPLING_CHUNK_SIZE; 
  onembed[0]  = RESAMPLING_CHUNK_SIZE/2+1;
  batch[1]    = RESAMPLING_BATCH;
  repeat[1]   = (BENG_FRAMES_PER_BLOCK-1)*2*BENG_CHANNELS_*BENG_SNAPSHOTS/RESAMPLING_CHUNK_SIZE/RESAMPLING_BATCH;
  if( cufftPlanMany(&(cufft_plan[1]), 1, &(fft_size[1]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_R2C,batch[1]) != CUFFT_SUCCESS)
  {
    hashpipe_error(__FILE__, "CUFFT error: plan 1 creation failed");
    state = STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cufftGetSize(cufft_plan[1],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 1 is %dx%dx%d\n", repeat[1],batch[1],fft_size[1]);
  fprintf(stdout,"GPU_DEBUG : plan 1 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  

 /*
 * FFT transforming complex spectrum into resampled time series simultaneously
 * trimming the SWARM guard bands (150 MHz -- 1174 MHz).
 * Input data has dimension EXPANSIONS_FACTOR
 */
  skip_chan = (int)(150e6 / (SWARM_RATE / RESAMPLING_CHUNK_SIZE));
  fft_size[2] = RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR;
  inembed[0]  = RESAMPLING_CHUNK_SIZE/2+1; 
  onembed[0]  = RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR;
  batch[2]    = batch[1];
  repeat[2]   = repeat[1];
  if( cufftPlanMany(&(cufft_plan[2]), 1, &(fft_size[2]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_C2R,batch[2]) != CUFFT_SUCCESS )
  {
    hashpipe_error(__FILE__, "CUFFT error: plan 2 creation failed");
    state = STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cufftGetSize(cufft_plan[2],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 2 is %dx%dx%d\n", repeat[2],batch[2],fft_size[2]);
  fprintf(stdout,"GPU_DEBUG : masking first %d channels \n",skip_chan);
  fprintf(stdout,"GPU_DEBUG : plan 2 worksize: %u\n",(unsigned) workSize[0]);
  fprintf(stdout,"GPU_DEBUG : channel mask starts at %d \n", skip_chan);
  reportDeviceMemInfo();
#endif  

  while (run_threads()) { // hashpipe wants us to keep running

    switch(state) {

    case STATE_ERROR:

      {

	// set status to show we're in error
	aphids_set(&aphids_ctx, "status", "error");

	// do nothing, wait to be killed
	sleep(1);

	break;

      }

    case STATE_INIT:

      {

	// set status to show we're going to run GPU code
	aphids_set(&aphids_ctx, "status", "processing data on GPU");

	// and set our next state
	state = STATE_PROC;

      }

    case STATE_PROC:

      {

	// read from the input buffer first
	while ((rv = hashpipe_databuf_wait_filled((hashpipe_databuf_t *)db_in, index_in)) != HASHPIPE_OK) {

	  if (rv == HASHPIPE_TIMEOUT) { // index_in is not ready
	    aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe input databuf timeout");

	    // need to check run_threads here again
	    if (run_threads())
	      continue;
	    else
	      break;

	  } else { // any other return value is an error

	    // raise an error and exit thread
	    hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
	    state = STATE_ERROR;
	    break;

	  }

	}

	// grab the data at this index_in
	this_vdif_packet_block = (vdif_in_packet_block_t)db_in->blocks[index_in];

	// let hashpipe know we're done with the buffer (for now)
	hashpipe_databuf_set_free((hashpipe_databuf_t *)db_in, index_in);

	// update the index_in modulo the maximum buffer depth
	index_in = (index_in + 1) % db_in->header.n_block;

	// now, write to the output buffer
	while ((rv = hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, index_out)) != HASHPIPE_OK) {

	  if (rv == HASHPIPE_TIMEOUT) { // index_out is not ready
	    aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe output databuf timeout");
	    continue;

	  } else { // any other return value is an error

	    // raise an error and exit thread
	    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
	    state = STATE_ERROR;
	    break;

	  }

	}

	/* GPU processing code ---> */

	// transform SWARM spectra into timeseries
#ifdef GPU_COMPUTE

#ifdef GPU_DEBUG
	cudaEventRecord(tic,stream);
#endif
	for (i = 0; i < repeat[0]; ++i) {
		//cufft_status = cufftExecC2R(plan,
		cufft_status = cufftExecC2R(cufft_plan[0],
			gpu_a + i*batch[0]*UNPACKED_BENG_CHANNELS, 
			gpu_b + i*batch[0]*(2*BENG_CHANNELS_));
#ifdef GPU_DEBUG
		if (cufft_status != CUFFT_SUCCESS){
		    hashpipe_error(__FILE__, "CUFFT error: plan 0 execution failed");
		    state = STATE_ERROR;
		    break;
 		}
#endif // GPU_DEBUG
	}
#ifdef GPU_DEBUG
	cudaEventRecord(toc,stream);
	cudaEventSynchronize(toc);
	if (aphids_ctx.iters % APHIDS_UPDATE_EVERY == 0) {
		cudaEventElapsedTime(&elapsedTime,tic,toc);
		aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "plan 0 cufft took %f ms [%f Gbps]",
			elapsedTime,  2*repeat[0]*batch[0]*fft_size[0] / (elapsedTime * 1e-3) /1e9 /2);
		fprintf(stdout,"GPU_DEBUG : plan 0 took %f ms [%f Gbps] \n", 
			elapsedTime,  2*repeat[0]*batch[0]*fft_size[0] / (elapsedTime * 1e-3) /1e9 /2);
		fflush(stdout);
	}
#endif // GPU_DEBUG

#ifdef GPU_DEBUG
	cudaEventRecord(tic,stream);
#endif
	// transform timeseries into reconfigured spectra
	for (i = 0; i < repeat[1]; ++i) {
		cufft_status = cufftExecR2C(cufft_plan[1],
			gpu_b + i*batch[1]*RESAMPLING_CHUNK_SIZE,
			gpu_a + i*batch[1]*(RESAMPLING_CHUNK_SIZE/2+1));
#ifdef GPU_DEBUG
		if (cufft_status != CUFFT_SUCCESS){
		    hashpipe_error(__FILE__, "CUFFT error: plan 1 execution failed");
		    state = STATE_ERROR;
		    break;
 		}
#endif // GPU_DEBUG
	}
#ifdef GPU_DEBUG
	cudaEventRecord(toc,stream);
	cudaEventSynchronize(toc);
	if (aphids_ctx.iters % APHIDS_UPDATE_EVERY == 0) {
		cudaEventElapsedTime(&elapsedTime,tic,toc);
		aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "plan 1 cufft took %f ms [%f Gbps]",
			elapsedTime,  2*repeat[1]*batch[1]*fft_size[1] / (elapsedTime * 1e-3) /1e9 /2);
		fprintf(stdout,"GPU_DEBUG : plan 1 took %f ms [%f Gbps] \n", 
			elapsedTime,  2*repeat[1]*batch[1]*fft_size[1] / (elapsedTime * 1e-3) /1e9 /2);
		fflush(stdout);
	}
#endif // GPU_DEBUG

#ifdef GPU_DEBUG
	cudaEventRecord(tic,stream);
#endif
	// mask and transform reconfigured spectra into resampled timeseries 
	for (i = 0; i < repeat[2]; ++i) {
		cufft_status = cufftExecC2R(cufft_plan[2],
			gpu_a + i*batch[2]*(RESAMPLING_CHUNK_SIZE/2+1) + skip_chan,
			gpu_b + i*batch[2]*(RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR));
#ifdef GPU_DEBUG
		if (cufft_status != CUFFT_SUCCESS){
		    hashpipe_error(__FILE__, "CUFFT error: plan 2 execution failed");
		    state = STATE_ERROR;
		    break;
 		}
#endif // GPU_DEBUG
	}
#ifdef GPU_DEBUG
	cudaEventRecord(toc,stream);
	cudaEventSynchronize(toc);
	if (aphids_ctx.iters % APHIDS_UPDATE_EVERY == 0) {
		cudaEventElapsedTime(&elapsedTime,tic,toc);
		aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "plan 2 cufft took %f ms [%f Gbps]",
			elapsedTime,  2*repeat[2]*batch[2]*fft_size[2] / (elapsedTime * 1e-3) /1e9 /2);
		fprintf(stdout,"GPU_DEBUG : plan 2 took %f ms [%f Gbps] \n", 
			elapsedTime,  2*repeat[2]*batch[2]*fft_size[2] / (elapsedTime * 1e-3) /1e9 /2);
		fflush(stdout);
	}
#endif // GPU_DEBUG
#endif // GPU_COMPUTE

	cudaStreamSynchronize(stream);

	// since the output buffer is a different size,
	// we have to manually fill it with the input data
	for (i = 0; i < VDIF_IN_PKTS_PER_BLOCK; i++) {
	  memcpy(&db_out->blocks[index_out].packets[i], &this_vdif_packet_block.packets[i], sizeof(vdif_in_packet_t));
	}

	/* <--- GPU processing code */

	// let hashpipe know we're done with the buffer (for now)
	hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, index_out);

	// update the index_out modulo the maximum buffer depth
	index_out = (index_out + 1) % db_out->header.n_block;

	// update aphids statistics
	aphids_update(&aphids_ctx);

	break;
      } // case STATE_PROC

    } // switch(state)

  } // end while(run_threads())

  /* GPU clean-up code */
  cudaFree(gpu_a);
  cudaFree(gpu_b);
  cudaEventDestroy(tic);
  cudaEventDestroy(toc);
  for (i=0; i<3; ++i){
	  cufftDestroy(cufft_plan[i]);
  }
  cudaStreamDestroy(stream);

  // destroy aphids context and exit
  aphids_destroy(&aphids_ctx);

  return NULL;
}

static hashpipe_thread_desc_t vdif_inout_gpu_thread = {
 name: "vdif_inout_gpu_thread",
 skey: "VDIFIO",
 init: NULL,
 run:  run_method,
 ibuf_desc: {vdif_in_databuf_create},
 obuf_desc: {vdif_out_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&vdif_inout_gpu_thread);
}
