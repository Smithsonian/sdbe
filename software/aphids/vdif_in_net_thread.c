#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/time.h>

#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "sgcomm_net.h"

#include "vdif_in_databuf.h"

#define STATE_ERROR       -1
#define STATE_IDLE         0
#define STATE_RECEIVE      1
#define STATE_TRANSFER     2
#define STATE_DONE        99

#define RX_HOST "localhost" //"192.168.10.10" // hamster IP on same 10GbE network as Mark6-4015
#define RX_PORT ((uint16_t)61234) // port to use for incoming VDIF packets

static void *run_method(hashpipe_thread_args_t * args) {
	
	int rv = 0;
	int index = 0;
	vdif_in_databuf_t *db_out = (vdif_in_databuf_t *)args->obuf;
	aphids_context_t aphids_ctx;
	int state = STATE_IDLE;
	//~ char in_file[256];
	//~ char status[256];
	
	// for logging
	#define TMP_MSG_LEN 80
	char tmp_msg[TMP_MSG_LEN];
	
	// for sgcomm_net.h interface
	int sockfd_server = 0;
	int sockfd_data = 0;
	void *local_buf = NULL;
	ssize_t nmem = 0;
	ssize_t size = 0;
	int timeouts = 5;
	
	// keeping track of received frames
	int offset_out = 0;
	int offset_in = 0;
	int copy_count = 0;
	int rx_count = 0;
	//~ int64_t first_b_count = -1;
	//~ int64_t this_b_count = -1;
	//~ vdif_header_t *vdif_hdr;
	
	// initialize the aphids context
	rv = aphids_init(&aphids_ctx, args);
	if (rv != APHIDS_OK) {
		hashpipe_error(__FUNCTION__, "error initializing APHIDS context");
		return NULL;
	}
	
	while (run_threads()) { // hashpipe wants us to keep running
		switch(state) {
			case STATE_ERROR: {
				// set status to show we're in error
				aphids_set(&aphids_ctx, "status", "error");
				// do nothing, wait to be killed
				sleep(1);
				break; // switch(state)
			}
			case STATE_IDLE: {
				// set status to show we're setting up server
				aphids_set(&aphids_ctx, "status", "starting server");
				aphids_set(&aphids_ctx,"status:net","none");
				// create socket and bind
				sockfd_server = make_socket_bind_listen(RX_HOST, RX_PORT);
				if (sockfd_server < 0) {
					aphids_set(&aphids_ctx,"status:net","error-on-bind");
					hashpipe_error(__FUNCTION__,"error binding socket");
					state = STATE_ERROR;
					break; // switch(state)
				}
				aphids_set(&aphids_ctx,"status:net","bound");
				// accept an incoming connection
				do {
					sockfd_data = accept_connection(sockfd_server);
					if (sockfd_data < 0) {
						if (sockfd_data == ERR_NET_TIMEOUT) {
							snprintf(tmp_msg,TMP_MSG_LEN,"timeout-on-connect (%d)",timeouts);
							aphids_set(&aphids_ctx,"status:net",tmp_msg);
							continue; // do ... while(--timeouts >0);
						} else {
							aphids_set(&aphids_ctx,"status:net","error-on-connect");
							hashpipe_error(__FUNCTION__,"error accepting connection");
							state = STATE_ERROR;
							break; // do ... while(--timeouts >0);
						}
					} else {
						aphids_set(&aphids_ctx,"status:net","connected");
						break; // do ... while(--timeouts > 0);
					}
				} while(--timeouts > 0);
				if (timeouts <= 0) {
					aphids_set(&aphids_ctx,"status:net","error-on-connect");
					hashpipe_error(__FUNCTION__,"error accepting connection");
					state = STATE_ERROR;
				}
				if (state == STATE_ERROR) {
					break; // switch(state)
				}
				aphids_set(&aphids_ctx, "status", "ready to receive data");
				state = STATE_RECEIVE;
				break; // switch(state)
			} // case STATE_IDLE
			case STATE_RECEIVE: {
				//~ snprintf(tmp_msg,TMP_MSG_LEN,"receive?(%d)",rx_count);
				//~ aphids_set(&aphids_ctx,"status:net",tmp_msg);
				// read batch of VDIF packets into local buffer
				//   rv is the expected number of frames
				//   nmem is the actual number of frames received
				//   size is the size of frames received
				rv = rx_frames(sockfd_data, &local_buf, &nmem, &size);
				if (rv < 0) {
					aphids_set(&aphids_ctx,"status:net","error-on-receive");
					hashpipe_error(__FUNCTION__,"error receiving data");
					state = STATE_ERROR;
					break; // switch(state)
				} else if (rv == 0) {
					// this means end-of-transmission
					state = STATE_DONE;
					break; // switch(state)
				}
				if (size != sizeof(vdif_in_packet_t)) {
					// free buffer, it does not contain useful data
					free(local_buf);
					hashpipe_warn(__FUNCTION__,"warning, packet size different than expected, ignoring batch");
					break; // while(run_threads()) ...
				}
				//~ snprintf(tmp_msg,TMP_MSG_LEN,"receive!(%d) (%lu/%d x %lu)",rx_count,nmem,rv,size);
				//~ aphids_set(&aphids_ctx,"status:net",tmp_msg);
				rx_count++;
				state = STATE_TRANSFER;
				break; // switch(state)
			} // case STATE_RECEIVE
			case STATE_TRANSFER: {
				/* offset_in < nmem means there is data in local_buf, so
				 * we can continue locally to transfer data to shared 
				 * buffer */
				while (offset_in < nmem) {
					// wait for buffer index to be ready
					while ((rv = hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, index)) != HASHPIPE_OK) {
						if (rv == HASHPIPE_TIMEOUT) { // index is not ready
							aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe databuf timeout");
							continue; // while (hashpipe_databuf_wait_free(...) ... )
						} else { // any other return value is an error
							// raise an error and exit thread
							hashpipe_error(__FUNCTION__, "error waiting for free databuf");
							state = STATE_ERROR;
							break; // while (hashpipe_databuf_wait_free(...) ... )
						}
					}
					//~ snprintf(tmp_msg,TMP_MSG_LEN,"status:out-buf(%d)",index);
					//~ aphids_set(&aphids_ctx,tmp_msg,"free");
					// calculate how much data we can transfer
					copy_count = (nmem-offset_in) < (VDIF_IN_PKTS_PER_BLOCK-offset_out) ? (nmem-offset_in) : (VDIF_IN_PKTS_PER_BLOCK-offset_out);
					//~ snprintf(tmp_msg,TMP_MSG_LEN,"%d",copy_count);
					//~ aphids_set(&aphids_ctx,"vars:copy_count",tmp_msg);
					memcpy(&db_out->blocks[index]+offset_out*sizeof(vdif_in_packet_t), local_buf+offset_in*sizeof(vdif_in_packet_t), copy_count*sizeof(vdif_in_packet_t));
					// update offset into local and shared buffers
					offset_in += copy_count;
					//~ snprintf(tmp_msg,TMP_MSG_LEN,"%d",offset_in);
					//~ aphids_set(&aphids_ctx,"vars:offset_in",tmp_msg);
					offset_out += copy_count;
					//~ snprintf(tmp_msg,TMP_MSG_LEN,"%d",offset_out);
					//~ aphids_set(&aphids_ctx,"vars:offset_out",tmp_msg);
					//~ usleep(10000);
					if (offset_out == VDIF_IN_PKTS_PER_BLOCK) {
						// shared buffer block full, reset offset
						offset_out = 0;
						// let hashpipe know we're done with the buffer (for now)
						hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, index);
						// update the index modulo the maximum buffer depth
						index = (index + 1) % db_out->header.n_block;
						// update aphids statistics
						aphids_update(&aphids_ctx);
						/* Continue to next block. Leave this continue
						 * statement here, in case we add something 
						 * below the if() {...} that should not happen 
						 * from here */
						continue; // while (offset_in < nmem)
					}
				}
				/* Done with local buffer, reset the offset and free the
				 * resource */
				offset_in = 0;
				if (local_buf != NULL) {
					free(local_buf);
				}
				
				/* If we're here, it means that the local buffer has
				 * been depleted, use Rx to replenish */
				state = STATE_RECEIVE;
				break; // switch(state)
			} // case STATE_TRANSFER
			case STATE_DONE: {
				// set status to show we're done
				aphids_set(&aphids_ctx, "status", "done");
				// do nothing, wait to be killed
				sleep(1);
				break; // switch(state)
			}
		} // switch(state)
	} // end while(run_threads())
	
	aphids_set(&aphids_ctx,"status:thread","stopping");
	
	// close sockets
	close(sockfd_server);
	close(sockfd_data);
	
	aphids_set(&aphids_ctx,"status:net","disconnected");
	
	aphids_set(&aphids_ctx,"status:thread","done");
	
	// destroy aphids context and exit
	aphids_destroy(&aphids_ctx);
	
	return NULL;
}

static hashpipe_thread_desc_t vdif_in_net_thread = {
	name: "vdif_in_net_thread",
	skey: "VDIFIN",
	init: NULL,
	run:  run_method,
	ibuf_desc: {NULL},
	obuf_desc: {vdif_in_databuf_create}
};

static __attribute__((constructor)) void ctor() {
	register_hashpipe_thread(&vdif_in_net_thread);
}
