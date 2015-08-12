#include <stdio.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/time.h>

#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_out_databuf.h"

#define STATE_ERROR    -1
#define STATE_INIT      0
#define STATE_DROP      1


static void *run_method(hashpipe_thread_args_t * args) {

  int rv = 0;
  int index = 0;
  quantized_storage_t this_quantized_storage;
  vdif_out_databuf_t *db_in = (vdif_out_databuf_t *)args->ibuf;
  aphids_context_t aphids_ctx;
  int state = STATE_INIT;

  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
    return NULL;
  }

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

	// set status to show we're dropping data
	aphids_set(&aphids_ctx, "status", "dropping data");

	// and set our next state
	state = STATE_DROP;

      }

    case STATE_DROP:

      {

	while ((rv = hashpipe_databuf_wait_filled((hashpipe_databuf_t *)db_in, index)) != HASHPIPE_OK) {

	  if (rv == HASHPIPE_TIMEOUT) { // index is not ready
	    aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe output databuf timeout");
	    //~ fprintf(stderr,"%s:%d:timeout\n",__FILE__,__LINE__);
	    continue;

	  } else { // any other return value is an error

	    // raise an error and exit thread
	    hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
	    //~ fprintf(stderr,"%s:%d: Hashpipe error\n",__FILE__,__LINE__);
	    state = STATE_ERROR;
	    break;

	  }

	}

	// grab the data at this index
	this_quantized_storage = (quantized_storage_t)db_in->blocks[index];

	// let hashpipe know we're done with the buffer (for now)
	hashpipe_databuf_set_free((hashpipe_databuf_t *)db_in, index);

	// update the index modulo the maximum buffer depth
	index = (index + 1) % db_in->header.n_block;

	// update aphids statistics
	aphids_update(&aphids_ctx);

	break;
      } // case STATE_DROP

    } // switch(state)

  } // end while(run_threads())

  // destroy aphids context and exit
  aphids_destroy(&aphids_ctx);

  return NULL;
}

static hashpipe_thread_desc_t vdif_out_null_thread = {
 name: "vdif_out_null_thread",
 skey: "VDIFOUT",
 init: NULL,
 run:  run_method,
 ibuf_desc: {vdif_out_databuf_create},
 obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&vdif_out_null_thread);
}
