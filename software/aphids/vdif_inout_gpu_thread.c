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

#define STATE_ERROR    -1
#define STATE_INIT      0
#define STATE_PROC      1
//********************************************** B-engine bookkeeping //
#define STATE_VDIF_IN  10
// B-engine bookkeeping **********************************************//

//********************************************** B-engine bookkeeping //
// B-engine bookkeeping **********************************************//

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
  
//********************************************** B-engine bookkeeping //
	#define NUM_GPU 4
	int64_t b_first = -1;
	beng_group_completion_buffer_t bgc_buf;
	beng_group_vdif_buffer_t *bgv_buf_cpu[BENG_GROUPS_IN_BUFFER];
	beng_group_vdif_buffer_t *bgv_buf_gpu[BENG_GROUPS_IN_BUFFER];
// B-engine bookkeeping **********************************************//
  
  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
    return NULL;
  }

  /* GPU initialization code */

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

//********************************************** B-engine bookkeeping //
for (ii=0; ii<BENG_GROUPS_IN_BUFFER; ii++) {
	// allocate memory on CPU
	bgv_buf_cpu[ii] = (beng_group_vdif_buffer_t *)malloc(sizeof(beng_group_vdif_buffer_t));
#if PLATFORM == PLATFORM_CPU
	// allocate memory on GPU
	bgv_buf_gpu[ii] = (beng_group_vdif_buffer_t *)malloc(sizeof(beng_group_vdif_buffer_t));
#elif PLATFORM == PLATFORM_GPU
	// select GPU
	device_id = ii % NUM_GPU;
	err_gpu = cudaSetDevice(device_id);
	if (err_gpu != cudaSuccess) {
		hashpipe_error(__FUNCTION__, "error selecting GPU device");
		state = STATE_ERROR;
		break; // for (ii=0; ...) ...
	}
	// allocate memory on GPU
	err_gpu = cudaMalloc((void **)&bgv_buf_gpu[ii], sizeof(beng_group_vdif_buffer_t));
	if (err_gpu != cudaSuccess) {
		hashpipe_error(__FUNCTION__, "error allocating memory on GPU device");
		state = STATE_ERROR;
		break; // for (ii=0; ...) ...
	}
}
// B-engine bookkeeping **********************************************//

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

//********************************************** B-engine bookkeeping //
// On first frame received, initialize all completion buffers
if (b_first == -1) {
	b_first = get_packet_b_count(&this_vdif_packet_block->packets[0].header);
	// from there on range starts with end value for previous range
	for (ii=0; ii<VDIF_UPLOAD_TO_GPU_BUFFER_SIZE; ii++) {
		// initialize output databuffer blocks
		init_beng_group(bgc, beng_group_vdif_buffer_t *bgv_buf, int64_t b_start);
		init_block(&bgc_buf, beng_vdif_buf_gpu, b_first+1+(BENG_FRAMES_PER_BLOCK-1)*ii);
	}
}
// B-engine bookkeeping **********************************************//


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
