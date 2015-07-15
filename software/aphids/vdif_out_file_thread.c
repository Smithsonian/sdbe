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
  int fd, bytes_written = 0, blocks_written = 0;
  vdif_in_databuf_t *db_in = (vdif_in_databuf_t *)args->ibuf;
  aphids_context_t aphids_ctx;

  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
    pthread_exit(NULL);
  }

  // open the VDIF output file
  fd = open("sample_out.vdif", O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
  if (fd < 0) {
    hashpipe_error(__FUNCTION__, "error opening VDIF file");
    pthread_exit(NULL);
  }

  while (run_threads()) { // hashpipe wants us to keep running

    while ((rv = hashpipe_databuf_wait_filled((hashpipe_databuf_t *)db_in, index)) != HASHPIPE_OK) {

      if (rv == HASHPIPE_TIMEOUT) { // index is not ready
	continue;

      } else { // any other return value is an error

	// raise an error and exit thread
	hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
	pthread_exit(NULL);
	break;

      }

    }

    // write one vdif packet block from the buffer at this index
    while (bytes_written < sizeof(vdif_in_packet_block_t)) {
      bytes_written += write(fd, &db_in->blocks[index] + bytes_written, sizeof(vdif_in_packet_block_t) - bytes_written);
    }

    // reset bytes_written
    bytes_written = 0;

    // increment blocks_written
    blocks_written++;

    // we've written everything, start over
    if (blocks_written == 1024) {
      aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "reached writing limit");
      lseek(fd, 0, SEEK_SET);
      blocks_written = 0;
    }

    // let hashpipe know we're done with the buffer (for now)
    hashpipe_databuf_set_free((hashpipe_databuf_t *)db_in, index);

    // update the index modulo the maximum buffer depth
    index = (index + 1) % db_in->header.n_block;

    // update aphids statistics
    aphids_update(&aphids_ctx);

  } // end while(run_threads())

  // close the VDIF file
  close(fd);

  // destroy aphids context and exit
  aphids_destroy(&aphids_ctx);

  return NULL;
}

static hashpipe_thread_desc_t vdif_out_file_thread = {
 name: "vdif_out_file_thread",
 skey: "VDIFOUT",
 init: NULL,
 run:  run_method,
 ibuf_desc: {vdif_in_databuf_create},
 obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&vdif_out_file_thread);
}
