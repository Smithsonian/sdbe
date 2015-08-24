#include <stdio.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/time.h>

#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"

#define STATE_ERROR    -1
#define STATE_INIT      0
#define STATE_GEN_NULL  1


static void *run_method(hashpipe_thread_args_t * args) {

  int rv = 0;
  int index = 0;
  beng_group_completion_t null_beng_group_completion = {};
  vdif_in_databuf_t *db_out = (vdif_in_databuf_t *)args->obuf;
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

	// set status to show we're going to generate
	aphids_set(&aphids_ctx, "status", "generating nulls");

	// and set our next state
	state = STATE_GEN_NULL;

      }

    case STATE_GEN_NULL:

      {

	while ((rv = hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, index)) != HASHPIPE_OK) {

	  if (rv == HASHPIPE_TIMEOUT) { // index is not ready
	    aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe databuf timeout");
	    continue;

	  } else { // any other return value is an error

	    // raise an error and exit thread
	    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
	    state = STATE_ERROR;
	    break;

	  }

	}

	// update the data at this index
	db_out->bgc[index] = null_beng_group_completion;

	// let hashpipe know we're done with the buffer (for now)
	hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, index);

	// update the index modulo the maximum buffer depth
	index = (index + 1) % db_out->header.n_block;

	// update aphids statistics
	aphids_update(&aphids_ctx);

	break;
      } // case STATE_GEN_NULL

    } // switch(state)

  } // end while(run_threads())

  // destroy aphids context and exit
  aphids_destroy(&aphids_ctx);

  return NULL;
}

static hashpipe_thread_desc_t vdif_in_null_thread = {
 name: "vdif_in_null_thread",
 skey: "VDIFIN",
 init: NULL,
 run:  run_method,
 ibuf_desc: {NULL},
 obuf_desc: {vdif_in_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&vdif_in_null_thread);
}
