#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/time.h>

#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"

#define STATE_ERROR    -1
#define STATE_IDLE      0
#define STATE_READ      1


static void *run_method(hashpipe_thread_args_t * args) {

  int rv = 0;
  int index = 0;
  int fd, bytes_read = 0;
  vdif_in_databuf_t *db_out = (vdif_in_databuf_t *)args->obuf;
  aphids_context_t aphids_ctx;
  int state = STATE_IDLE;
  char in_file[256];
  char status[256];

  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error initializing APHIDS context");
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

    case STATE_IDLE:

      {

	// set status to show we're waiting for a file
	aphids_set(&aphids_ctx, "status", "waiting for input file");

	// pop a file to read from our in_files queue
	rv = aphids_pop(&aphids_ctx, "in_files", in_file, 1);
	if (rv < 0) {
	  hashpipe_error(__FUNCTION__, "error popping from in_files");
	  state = STATE_ERROR;
	  break;
	}

	// make sure it's not a timeout
	if (rv != APHIDS_POP_TIMEOUT) {

	  // open the file for reading
	  fd = open(in_file, O_RDONLY);
	  if (fd < 0) {
	    hashpipe_error(__FUNCTION__, "error opening VDIF file");
	    state = STATE_ERROR;
	    break;
	  }

	  // set our status and go
	  sprintf(status, "reading (%s)", in_file);
	  aphids_set(&aphids_ctx, "status", status);

	  // change state
	  state = STATE_READ;

	}

	break;
      } // case STATE_IDLE

    case STATE_READ:

      {

	// wait for buffer index to be ready
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

	// read one vdif packet block and write it to the index
	while (bytes_read < sizeof(vdif_in_packet_block_t)) {
	  bytes_read += read(fd, &db_out->blocks[index] + bytes_read, sizeof(vdif_in_packet_block_t) - bytes_read);

	  if (bytes_read == 0) {

	    // we've reached EOF
	    aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "reached end-of-file");

	    // close the VDIF file
	    close(fd);

	    // and change states
	    state = STATE_IDLE;
	  }

	}

	// reset bytes_read
	bytes_read = 0;

	// let hashpipe know we're done with the buffer (for now)
	hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, index);

	// update the index modulo the maximum buffer depth
	index = (index + 1) % db_out->header.n_block;

	// update aphids statistics
	aphids_update(&aphids_ctx);

	break;
      } // case STATE_READ

    } // switch(state)

  } // end while(run_threads())

  // close the VDIF file
  close(fd);

  // destroy aphids context and exit
  aphids_destroy(&aphids_ctx);

  return NULL;
}

static hashpipe_thread_desc_t vdif_in_file_thread = {
 name: "vdif_in_file_thread",
 skey: "VDIFIN",
 init: NULL,
 run:  run_method,
 ibuf_desc: {NULL},
 obuf_desc: {vdif_in_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&vdif_in_file_thread);
}
