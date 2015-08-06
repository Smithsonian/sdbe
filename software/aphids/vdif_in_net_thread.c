#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/time.h>

#ifndef STANDALONE_TEST
#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"
#endif // STANDALONE_TEST

#include "sgcomm_net.h"

#include "vdif_in_databuf.h"

// compilation options, for testing and debugging only
#define PLATFORM_CPU 0
#define PLATFORM_GPU 1
#define PLATFORM PLATFORM_CPU
#if PLATFORM == PLATFORM_GPU
	#define NUM_GPU 4
	#include <cuda_runtime.h>
#endif


#define STATE_ERROR       -1
#define STATE_IDLE         0
#define STATE_PROCESS      1
#define STATE_DONE        99

#define RX_HOST "localhost" //"192.168.10.10" // hamster IP on same 10GbE network as Mark6-4015
#define RX_PORT ((uint16_t)12345) // port to use for incoming VDIF packets

// VDIF packet stores associated with each group of B-engine frames
beng_group_vdif_buffer_t *bgv_buf_cpu[BENG_GROUPS_IN_BUFFER];
beng_group_vdif_buffer_t *bgv_buf_gpu[BENG_GROUPS_IN_BUFFER];

static int init_method(
#ifndef STANDALONE_TEST
hashpipe_thread_args_t *args
#endif // STANDALONE_TEST
) {
	int ii;
	//////////////////////////////////////////// B-engine bookkeeping //
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
			printf("%s:%d: Error-state due to cudaSetDevice\n",__FILE__,__LINE__);
			state = STATE_ERROR;
			break; // for (ii=0; ...) ...
		}
		// allocate memory on GPU
		err_gpu = cudaMalloc((void **)&bgv_buf_gpu[ii], sizeof(beng_group_vdif_buffer_t));
		if (err_gpu != cudaSuccess) {
			hashpipe_error(__FUNCTION__, "error allocating memory on GPU device");
			printf("%s:%d: Error-state due to cudaMalloc\n",__FILE__,__LINE__);
			state = STATE_ERROR;
			break; // for (ii=0; ...) ...
		}
	#endif // PLATFORM == PLATFORM_CPU
	}
	// B-engine bookkeeping ////////////////////////////////////////////
	
	// guess zero return means success?
	return 0;
}

static void *run_method(
#ifndef STANDALONE_TEST
	hashpipe_thread_args_t * args
#else
	vdif_in_databuf_t *db_out
#endif
) {
	
	// misc
	int ii = 0;
	int rv = 0;
	
	// aphids-on-hashpipe stuff
	int index_db_out = 0;
#ifndef STANDALONE_TEST
	vdif_in_databuf_t *db_out = (vdif_in_databuf_t *)args->obuf;
	aphids_context_t aphids_ctx;
#endif // STANDALONE_TEST
	int state = STATE_IDLE;
	
	// for logging
	#define TMP_MSG_LEN 80
	char tmp_msg[TMP_MSG_LEN];
	
	// for sgcomm_net.h interface
	int sockfd_server = 0;
	int sockfd_data = 0;
	ssize_t size = 0;
	int timeouts = 5;
	
	// I'm all about that VDIF 
	void *received_vdif_packets = NULL;
	ssize_t n_received_vdif_packets = 0;
	ssize_t index_received_vdif_packets = 0;
	
	// B-engine bookkeeping
	int64_t b_first = -1;
	vdif_in_databuf_t local_db_out;
	#define MAX_INDEX_LOOK_AHEAD 1
	int index_offset = 0;
	int start_copy = 0;
	char copy_in_progress_flags[BENG_GROUPS_IN_BUFFER] = { 0 };
	
#ifndef STANDALONE_TEST
	// initialize the aphids context
	rv = aphids_init(&aphids_ctx, args);
	if (rv != APHIDS_OK) {
		hashpipe_error(__FUNCTION__, "error initializing APHIDS context");
		return NULL;
	}
#endif // STANDALONE_TEST
	
	while (
#ifndef STANDALONE_TEST
	run_threads()
#else
	1
#endif // STANDALONE_TEST
	) { // hashpipe wants us to keep running
		switch(state) {






			case STATE_ERROR: {
#ifndef STANDALONE_TEST
				// set status to show we're in error
				aphids_set(&aphids_ctx, "status", "error");
#endif // STANDALONE_TEST
				// do nothing, wait to be killed
				sleep(1);
				break; // switch(state)
			}






			case STATE_IDLE: {
#ifndef STANDALONE_TEST
				// set status to show we're setting up server
				aphids_set(&aphids_ctx, "status", "starting server");
				aphids_set(&aphids_ctx,"status:net","none");
#endif // STANDALONE_TEST
				// create socket and bind
				sockfd_server = make_socket_bind_listen(RX_HOST, RX_PORT);
				if (sockfd_server < 0) {
#ifndef STANDALONE_TEST
					aphids_set(&aphids_ctx,"status:net","error-on-bind");
					hashpipe_error(__FUNCTION__,"error binding socket");
#endif // STANDALONE_TEST
					//~ fprintf(stderr,"%s:%d: Hashpipe error\n",__FILE__,__LINE__);
					//~ printf("%s:%d: Error-state due to make_socket_bind_listen\n",__FILE__,__LINE__);
					state = STATE_ERROR;
					break; // switch(state)
				}
#ifndef STANDALONE_TEST
				aphids_set(&aphids_ctx,"status:net","bound");
#endif // STANDALONE_TEST
				// accept an incoming connection
				do {
					sockfd_data = accept_connection(sockfd_server);
					if (sockfd_data < 0) {
						if (sockfd_data == ERR_NET_TIMEOUT) {
#ifndef STANDALONE_TEST
							snprintf(tmp_msg,TMP_MSG_LEN,"timeout-on-connect (%d)",timeouts);
							aphids_set(&aphids_ctx,"status:net",tmp_msg);
#endif // STANDALONE_TEST
							continue; // do ... while(--timeouts >0);
						} else {
#ifndef STANDALONE_TEST
							aphids_set(&aphids_ctx,"status:net","error-on-connect");
							hashpipe_error(__FUNCTION__,"error accepting connection");
#endif // STANDALONE_TEST
							//~ fprintf(stderr,"%s:%d: Hashpipe error\n",__FILE__,__LINE__);
							printf("%s:%d: Error-state due to accept_connection\n",__FILE__,__LINE__);
							state = STATE_ERROR;
							break; // do ... while(--timeouts >0);
						}
					} else {
#ifndef STANDALONE_TEST
						aphids_set(&aphids_ctx,"status:net","connected");
						//~ fprintf(stderr,"%s:%d: test connected\n",__FILE__,__LINE__);
#endif // STANDALONE_TEST
						break; // do ... while(--timeouts > 0);
					}
				} while(--timeouts > 0);
				if (timeouts <= 0) {
#ifndef STANDALONE_TEST
					aphids_set(&aphids_ctx,"status:net","error-on-connect");
					hashpipe_error(__FUNCTION__,"error accepting connection");
#endif // STANDALONE_TEST
					//~ fprintf(stderr,"%s:%d: Hashpipe error\n",__FILE__,__LINE__);
					//~ printf("%s:%d: Error-state due to timeout on connect\n",__FILE__,__LINE__);
					state = STATE_ERROR;
				}
				if (state == STATE_ERROR) {
					break; // switch(state)
				}
#ifndef STANDALONE_TEST
				aphids_set(&aphids_ctx, "status", "ready to receive data");
#endif // STANDALONE_TEST
				//~ fprintf(stderr,"%s:%d: status process\n",__FILE__,__LINE__);
				state = STATE_PROCESS;
				break; // switch(state)
			} // case STATE_IDLE






			case STATE_PROCESS: {
// TODO: Receive VDIF data:
//   if (number of frames received is zero)
//     receive frames
//     set the number of frames received
				if (n_received_vdif_packets == 0) {
					// read batch of VDIF packets into local buffer
					//   rv is the expected number of frames
					//   n_received_vdif_packets is the actual number of frames received
					//   size is the size of frames received
					rv = rx_frames(sockfd_data, &received_vdif_packets, &n_received_vdif_packets, &size);
					//~ fprintf(stderr,"%s:%d: received %d packets\n",__FILE__,__LINE__,n_received_vdif_packets);
					if (rv < 0) {
						if (rv == ERR_NET_TIMEOUT) {
#ifndef STANDALONE_TEST
							aphids_set(&aphids_ctx,"status:net","timeout, retry");
#endif // STANDALONE_TEST
							break;
						}
#ifndef STANDALONE_TEST
						aphids_set(&aphids_ctx,"status:net","error-on-receive");
						hashpipe_error(__FUNCTION__,"error receiving data");
#endif // STANDALONE_TEST
						//~ fprintf(stderr,"%s:%d: Hashpipe error\n",__FILE__,__LINE__);
						//~ printf("%s:%d: Error-state due to network error\n",__FILE__,__LINE__);
						state = STATE_ERROR;
						break; // switch(state)
					} else if (rv == 0) {
						// this means end-of-transmission, reset state 
						b_first = -1;
						// should probably go to STATE_IDLE, but for now
						// just use STATE_DONE
						state = STATE_DONE;
						break; // switch(state)
					}
					if (size != sizeof(vdif_in_packet_t)) {
						// free buffer, it does not contain useful data
						free(received_vdif_packets);
#ifndef STANDALONE_TEST
						hashpipe_warn(__FUNCTION__,"warning, packet size different than expected, ignoring batch");
#endif // STANDALONE_TEST
						break; // switch(state)
					}
					
					// On first frame received, initialize all completion buffers
					if (b_first == -1) {
						b_first = get_packet_b_count((vdif_in_header_t *)received_vdif_packets);
						//~ printf("b_first = %ld\n",b_first);
						// from there on range starts with end value for previous range
						for (ii=0; ii<BENG_GROUPS_IN_BUFFER; ii++) {
							// initialize output databuffer blocks
							init_beng_group(local_db_out.bgc+ii, bgv_buf_cpu[ii], bgv_buf_gpu[ii], b_first+1 + (BENG_FRAMES_PER_GROUP-1)*ii);
						}
					}
					// B-engine bookkeeping ////////////////////////////////////////////
					// reset index into received packets
					index_received_vdif_packets = 0;
					//~ fprintf(stderr,"%s:%d: done receiving\n",__FILE__,__LINE__);
				}
				
// TODO: Fill VDIF buffers:
//   get index offset
//   if (offset too large)
//     mark fill index buffer ready for transfer
//     increment fill index
//   else
//     insert into buffer
//     increment rx buffer index
//     if (rx buffer index passed last data)
//       free the local buffer
//       reset the rx buffer index
//     if (buffer at fill index full)
//       mark fill index buffer ready for transfer
				/* index_received_vdif_packets < n_received_vdif_packets means there is data in received_vdif_packets, so
				 * we can continue locally to transfer data to shared 
				 * buffer */
				while (index_received_vdif_packets < n_received_vdif_packets) {
					// get index offset
					index_offset = get_beng_group_index_offset(&local_db_out, index_db_out, (vdif_in_packet_t *)received_vdif_packets + index_received_vdif_packets);
					if (index_offset < 0) {
						// throw away these frames, they are from before
						// the range we're interested in
						index_received_vdif_packets++;
						continue;
					}
					if (index_offset > MAX_INDEX_LOOK_AHEAD) {
						// set transfer on non-filled unit
						start_copy = 1;
					} else {
						// insert VDIF packet: this copies VDIF to the 
						// local buffer, updates the frame counters and
						// flags, and returns the number of insertions
						insert_vdif_in_beng_group_buffer(&local_db_out, index_db_out, index_offset, (vdif_in_packet_t *)received_vdif_packets + index_received_vdif_packets);
						// we're done with this VDIF packet, increment 
						// index
						index_received_vdif_packets++;
						// check if B-engine group ready for transfer
						if (check_beng_group_complete(&local_db_out, index_db_out)) {
							// set transfer on filled unit
							start_copy = 1;
						}
					}
					// check if we should start copy
					if (start_copy) {
						break; // while(index_received_vdif_packets < n_received_vdif_packets)
					}
				}
				// check if we're done with received data
				if (index_received_vdif_packets == n_received_vdif_packets) {
					free(received_vdif_packets);
					n_received_vdif_packets = 0;
				}
				//~ fprintf(stderr,"%s:%d: done processing\n",__FILE__,__LINE__);
				
// TODO: Start asynchronous data transfer
//   wait on output buffer empty (hashpipe) at index
//   start copy of buffer ready
//   set copy-in-progress flag (in vector)
//   increment fill index
				if (start_copy) {
					//~ fprintf(stderr,"%s:%d: start copying\n",__FILE__,__LINE__);
#ifndef STANDALONE_TEST
					//~ printf("start_copy");
					// wait for buffer index to be ready, there may be
					// busy data at destination
					while ((rv = hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, index_db_out)) != HASHPIPE_OK) {
						if (rv == HASHPIPE_TIMEOUT) { // index is not ready
							aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe databuf timeout");
							continue; // while (hashpipe_databuf_wait_free(...) ... )
						} else { // any other return value is an error
							// raise an error and exit thread
							hashpipe_error(__FUNCTION__, "error waiting for free databuf");
							//~ fprintf(stderr,"%s:%d: Hashpipe error\n",__FILE__,__LINE__);
							//~ printf("%s:%d: Error-state due to wait free\n",__FILE__,__LINE__);
							state = STATE_ERROR;
							break; // while (hashpipe_databuf_wait_free(...) ... )
						}
					}
					//~ printf("%s:%d: output block %d free\n",__FILE__,__LINE__,index_db_out);
#endif // STANDALONE_TEST
					// now we know we're free to copy data; this starts
					// the transfer but returns asynchronously
					transfer_beng_group_to_gpu(&local_db_out, index_db_out);
					copy_in_progress_flags[index_db_out] = 1;
					// update the index modulo the maximum buffer depth
					index_db_out = (index_db_out + 1) % BENG_GROUPS_IN_BUFFER;
					// reset the start_copy flag
					start_copy = 0;
				}
				
// TODO: Check when copy is done
//   for each copy-in-progress
//     if (this copy is done)
//       copy the meta-data to output buffer (hashpipe)
//       mark output buffer filled (hashpipe at index)
//       clean-up and init local data / completion buffers
//       reset copy-in-progress flag (in vector)
				for (ii=0; ii<BENG_GROUPS_IN_BUFFER; ii++) {
					// check only those for which copy has started
					if (copy_in_progress_flags[ii]) {
						// if the copy is done
						if (check_transfer_complete(ii)) {
							//mark copy complete
							copy_in_progress_flags[ii] = 0;
							// copy metadata
							memcpy(&(db_out->bgc[ii]), &(local_db_out.bgc[ii]), sizeof(beng_group_completion_t));
							// let hashpipe know we're done with the buffer (for now)
#ifndef STANDALONE_TEST
							hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, ii);
							// update aphids statistics
							aphids_update(&aphids_ctx);
#endif // STANDALONE_TEST
							init_beng_group(local_db_out.bgc+ii, bgv_buf_cpu[ii], bgv_buf_gpu[ii], local_db_out.bgc[(ii + BENG_GROUPS_IN_BUFFER - 1) % BENG_GROUPS_IN_BUFFER].bfc[BENG_FRAMES_PER_GROUP-1].b);
						}
					}
				} // for
				break; // switch(state)
			} // case STATE_PROCESS:





			case STATE_DONE: {
#ifndef STANDALONE_TEST
				// set status to show we're done
				aphids_set(&aphids_ctx, "status", "done");
#endif // STANDALONE_TEST
				// do nothing, wait to be killed
				sleep(1);
				break; // switch(state)
			} // case STATE_DONE:






		} // switch(state)
	} // end while(run_threads())
	
#ifndef STANDALONE_TEST
	aphids_set(&aphids_ctx,"status:thread","stopping");
#endif // STANDALONE_TEST
	
	// close sockets
	close(sockfd_server);
	close(sockfd_data);
	
#ifndef STANDALONE_TEST
	aphids_set(&aphids_ctx,"status:net","disconnected");
	aphids_set(&aphids_ctx,"status:thread","done");
#endif // STANDALONE_TEST
	
#ifndef STANDALONE_TEST
	// destroy aphids context and exit
	aphids_destroy(&aphids_ctx);
#endif // STANDALONE_TEST
	
	return NULL;
}

#ifdef STANDALONE_TEST
int main(int argc, const char **argv) {
	vdif_in_databuf_t db_out;
	
	// initialize
	init_method();
	
	// run main loop
	run_method(&db_out);
	
	return 0;
}
#else 
static hashpipe_thread_desc_t vdif_in_net_thread = {
	name: "vdif_in_net_thread",
	skey: "VDIFIN",
	init: init_method,
	run: run_method,
	ibuf_desc: {NULL},
	obuf_desc: {vdif_in_databuf_create}
};

static __attribute__((constructor)) void ctor() {
	register_hashpipe_thread(&vdif_in_net_thread);
}
#endif // STANDALONE_TEST
