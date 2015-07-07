#include <sys/stat.h>
#include <fcntl.h>
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
  int index = 0;
  int fd, bytes_read = 0;
  vdif_in_databuf_t *db_out = (vdif_in_databuf_t *)args->obuf;
  aphids_context_t aphids_ctx;

  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
    pthread_exit(NULL);
  }

  // open the VDIF input file
  fd = open("sample_in.vdif", O_RDONLY);
  if (fd < 0) {
    hashpipe_error(__FUNCTION__, "error opening VDIF file");
    pthread_exit(NULL);
  }

  while (run_threads()) { // hashpipe wants us to keep running

    while ((rv = hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, index)) != HASHPIPE_OK) {

      if (rv == HASHPIPE_TIMEOUT) { // index is not ready
	aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe databuf timeout");
	continue;

      } else { // any other return value is an error

	// raise an error and exit thread
	hashpipe_error(__FUNCTION__, "error waiting for free databuf");
	pthread_exit(NULL);
	break;

      }

    }

    // read one vdif packet block and write it to the index
    while (bytes_read < sizeof(vdif_packet_block_t)) {
      bytes_read += read(fd, &db_out->blocks[index] + bytes_read, sizeof(vdif_packet_block_t) - bytes_read);
      if (bytes_read == 0) {
	aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "reached end-of-file");
	lseek(fd, 0, SEEK_SET);
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
