#include <stdio.h>
#include <syslog.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"


static void *run_method(hashpipe_thread_args_t * args) {

  int rv = 0;
  int index_in = 0;
  int index_out = 0;
  vdif_packet_t this_vdif_packet;
  vdif_in_databuf_t *db_in = (vdif_in_databuf_t *)args->ibuf;
  vdif_in_databuf_t *db_out = (vdif_in_databuf_t *)args->obuf;
  hashpipe_status_t st = args->st;
  const char * status_key = args->thread_desc->skey;
  aphids_context_t aphids_ctx;

  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
    pthread_exit(NULL);
  }


  // first log, just to say we're starting
  syslog(LOG_INFO, "%s[START]", args->thread_desc->name);

  // update our status
  hashpipe_status_lock_safe(&st);
  hputs(st.buf, status_key, "running");
  hashpipe_status_unlock_safe(&st);

  while (run_threads()) { // hashpipe wants us to keep running

    // read from the input buffer first
    while ((rv = hashpipe_databuf_wait_filled((hashpipe_databuf_t *)db_in, index_in)) != HASHPIPE_OK) {

      if (rv == HASHPIPE_TIMEOUT) { // index_in is not ready
	continue;

      } else { // any other return value is an error

	// raise an error and exit thread
	hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
	pthread_exit(NULL);
	break;

      }

    }

    // grab the data at this index_in
    this_vdif_packet = (vdif_packet_t)db_in->packets[index_in];

    // let hashpipe know we're done with the buffer (for now)
    hashpipe_databuf_set_free((hashpipe_databuf_t *)db_in, index_in);

    // update the index_in modulo the maximum buffer depth
    index_in = (index_in + 1) % db_in->header.n_block;

    // now, write to the output buffer
    while ((rv = hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, index_out)) != HASHPIPE_OK) {

      if (rv == HASHPIPE_TIMEOUT) { // index_out is not ready
	continue;

      } else { // any other return value is an error

	// raise an error and exit thread
	hashpipe_error(__FUNCTION__, "error waiting for free databuf");
	pthread_exit(NULL);
	break;

      }

    }

    // send the received packet right back out
    db_out->packets[index_out] = this_vdif_packet;

    // let hashpipe know we're done with the buffer (for now)
    hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, index_out);

    // update the index_out modulo the maximum buffer depth
    index_out = (index_out + 1) % db_out->header.n_block;

    // update aphids statistics
    aphids_update(&aphids_ctx);

    pthread_testcancel(); // check if thread has been canceled

  } // end while(run_threads())

  // update our status
  hashpipe_status_lock_safe(&st);
  hputs(st.buf, status_key, "stopping");
  hashpipe_status_unlock_safe(&st);

  // one last log, number iterations
  syslog(LOG_INFO, "%s[STOP]: iters=%d", args->thread_desc->name, aphids_ctx.iters);

  closelog(); // close logger

  return NULL;
}

static hashpipe_thread_desc_t vdif_inout_null_thread = {
 name: "vdif_inout_null_thread",
 skey: "VDIFIO",
 init: NULL,
 run:  run_method,
 ibuf_desc: {vdif_in_databuf_create},
 obuf_desc: {vdif_in_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&vdif_inout_null_thread);
}
