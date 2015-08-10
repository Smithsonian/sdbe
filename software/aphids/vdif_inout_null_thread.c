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
#define STATE_COPY      1


static void *run_method(hashpipe_thread_args_t * args) {

	int i = 0;
	int rv = 0;
	int index_in = 0;
	int index_out = 0;
	beng_group_completion_t this_beng_group_completion;
	vdif_in_databuf_t *db_in = (vdif_in_databuf_t *)args->ibuf;
	vdif_out_databuf_t *db_out = (vdif_out_databuf_t *)args->obuf;
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
			
			case STATE_ERROR: {
				// set status to show we're in error
				aphids_set(&aphids_ctx, "status", "error");
				// do nothing, wait to be killed
				sleep(1);
				break;
			}
			
			case STATE_INIT: {
				// set status to show we're going to generate
				aphids_set(&aphids_ctx, "status", "copying input to output");
				// and set our next state
				state = STATE_COPY;
			}
			
			case STATE_COPY: {
				
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
				//~ printf("%s:%d: input block %d filled\n",__FILE__,__LINE__,index_in);
				
				// grab the data at this index_in
				this_beng_group_completion = (beng_group_completion_t)db_in->bgc[index_in];
				
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
				//~ printf("%s:%d: output block %d free\n",__FILE__,__LINE__,index_out);
			
				// since the output buffer is a different size,
				// we have to manually fill it with the input data
				//~ for (i = 0; i < VDIF_IN_PKTS_PER_BLOCK; i++) {
				//~ memcpy(&db_out->blocks[index_out].packets[i], &this_vdif_packet_block.packets[i], sizeof(vdif_in_packet_t));
				//~ }
				
				// let hashpipe know we're done with the buffer (for now)
				hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, index_out);
				
				// update the index_out modulo the maximum buffer depth
				index_out = (index_out + 1) % db_out->header.n_block;
				
				// update aphids statistics
				aphids_update(&aphids_ctx);
			
				break;
			} // case STATE_COPY
		
		} // switch(state)
	
	} // end while(run_threads())
	
	// destroy aphids context and exit
	aphids_destroy(&aphids_ctx);
	
	return NULL;
}

static hashpipe_thread_desc_t vdif_inout_null_thread = {
	name: "vdif_inout_null_thread",
	skey: "VDIFIO",
	init: NULL,
	run:  run_method,
	ibuf_desc: {vdif_in_databuf_create},
	obuf_desc: {vdif_out_databuf_create}
};

static __attribute__((constructor)) void ctor() {
	register_hashpipe_thread(&vdif_inout_null_thread);
}
