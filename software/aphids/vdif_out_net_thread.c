#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#ifndef STANDALONE_TEST
#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"
#endif // STANDALONE_TEST

#include "sgcomm_net.h"

#include "vdif_out_databuf.h"

#define STATE_ERROR    -1
#define STATE_IDLE      0
#define STATE_PROCESS   1

#ifdef NET_HAMSTER // sets network parameters for hamster >> Mark6-4016
#define TX_HOST0 "10.2.2.32" // Mark6-4016 IP for receiving chan0 data
#define TX_PORT0 ((uint16_t)54323) // port for receiving chan0 data
#define TX_HOST1 "10.2.2.34" // Mark6-4016 IP for receiving chan1 data
#define TX_PORT1 ((uint16_t)54325) // port for receiving chan1 data
#else // sets network parameters for local test
#define TX_HOST0 "localhost"
#define TX_PORT0 ((uint16_t)61234)
#define TX_HOST1 "localhost"
#define TX_PORT1 ((uint16_t)61235)
#endif

static void *run_method(hashpipe_thread_args_t * args) {
	
	// misc
	int ii = 0;
	int jj = 0;
	int rv = 0;
	time_t current_time;
	
	// aphids-on-hashpipe stuff
	int index_db_in = 0;
	vdif_out_databuf_t qs_buf;
	vdif_out_databuf_t *db_in = (vdif_out_databuf_t *)args->ibuf;
	aphids_context_t aphids_ctx;
	int state = STATE_IDLE;
	
	// for logging
	#define TMP_MSG_LEN 80
	char tmp_msg[TMP_MSG_LEN];
	
	// for sgcomm_net.h interface
	int sockfd[VDIF_CHAN] = { 0 };
	int frames_sent[VDIF_CHAN] = { 0 };
	const char *tx_host[VDIF_CHAN] = { TX_HOST0, TX_HOST1 };
	const uint16_t tx_port[VDIF_CHAN] = { TX_PORT0, TX_PORT1 };
	//~ #define MAX_TX_TRIES 5
	//~ int tx_tries = 0;
	
	// I'm all about that VDIF
	vdif_out_data_group_t *vdg_buf_cpu[VDIF_OUT_BUFFER_SIZE];
	vdif_out_packet_group_t *vpg_buf_cpu[VDIF_OUT_BUFFER_SIZE];
	uint32_t df_num_insec, // wrapped incrementing counter
		ref_epoch, // fixed value
		secs_inre; // incrementing counter
	uint64_t edh_psn;
	
	// buffer leftover data for sub-VDIF-packet time-alignment
	int32_t *leftover_buffer[VDIF_CHAN];
	int32_t leftover_int32_t = 0;
	vdif_out_data_group_t *vdg_buf_cpu_time_aligned[VDIF_CHAN];
	
	// some time-alignment parameters
	int skip_int32_t;
	int skip_frames;
	int num_usable_packets;
	
	// data transfer bookkeeping
	int start_copy = 0;
	char copy_in_progress_flag = 0;
	
	// initialize the aphids context
	rv = aphids_init(&aphids_ctx, args);
	if (rv != APHIDS_OK) {
		hashpipe_error(__FUNCTION__, "error waiting for free databuf");
		return NULL;
	}
	
	// gather metadata for output VDIF header
	char *meta_str = malloc(sizeof(char)*16);
	if (aphids_get(&aphids_ctx, "vdif:station", meta_str) != APHIDS_OK) {
		fprintf(stdout,"%s:%s(%d): raw metadata failed to fetch station, using default\n",__FILE__,__FUNCTION__,__LINE__);
		strcpy(meta_str, "XX");
	}
        uint16_t meta_station = ((((uint16_t)meta_str[0])&0x00FF)<<8) | (((uint16_t)meta_str[1])&0x00FF);
	if (aphids_get(&aphids_ctx, "vdif:bdc", meta_str)  != APHIDS_OK) {
		fprintf(stdout,"%s:%s(%d): raw metadata failed to fetch bdc, using default\n",__FILE__,__FUNCTION__,__LINE__);
		strcpy(meta_str, "0");
	}
	uint8_t meta_bdc = 0x01 & ((uint8_t)atoi(meta_str));
	if (aphids_get(&aphids_ctx, "vdif:rx", meta_str)  != APHIDS_OK) {
		fprintf(stdout,"%s:%s(%d): raw metadata failed to fetch rx, using default\n",__FILE__,__FUNCTION__,__LINE__);
		strcpy(meta_str, "0");
	}
	uint8_t meta_rx = 0x01 & ((uint8_t)atoi(meta_str));
	fprintf(stdout,"%s:%s(%d): raw metadata is station=%d, bdc=%d, rx=%d\n",__FILE__,__FUNCTION__,__LINE__,meta_station,meta_bdc,meta_rx);
	free(meta_str);

	while (run_threads()) { // hashpipe wants us to keep running
	
		switch(state) {
			
			case STATE_ERROR: {
				// set status to show we're in error
				aphids_set(&aphids_ctx, "status", "error");
				// do nothing, wait to be killed
				sleep(1);
				break;
			} // case STATE_ERROR
			
			case STATE_IDLE: {
				// set status to show we're dropping data
				aphids_set(&aphids_ctx, "status", "dropping data");
				// TODO: allocate local memory buffers
				for (ii=0; ii<VDIF_OUT_BUFFER_SIZE; ii++) {
				//   * host memory for raw 2bit data blocks
					get_vdg_cpu_memory(&vdg_buf_cpu[ii], ii);
				//   * host memory for VDIF packetized data
					get_vpg_cpu_memory(&vpg_buf_cpu[ii], ii);
				}
				
				// TODO: setup network connection
				aphids_set(&aphids_ctx,"status:net","connecting");
				for (ii=0; ii<VDIF_CHAN; ii++) {
					sockfd[ii] = make_socket_connect(tx_host[ii],tx_port[ii]);
					fprintf(stdout,"%s:%s(%d): socket (%s:%d) opened with descriptor %d\n",__FILE__,__FUNCTION__,__LINE__,tx_host[ii],tx_port[ii],sockfd[ii]);
					if (sockfd[ii] < 0) {
						aphids_set(&aphids_ctx,"status:net","error-on-connect");
						hashpipe_error(__FUNCTION__,"error connecting");
						state = STATE_ERROR;
						break; // for ii over VDIF_CHAN
					}
				}
				if (state == STATE_ERROR) {
					break; // switch(state)
				}
				aphids_set(&aphids_ctx,"status:net","connected");
				
				// TODO: initialize data out
				//   * set VDIF header templates in buffer
				//   * set first round of timestamps
				for (ii=0; ii<VDIF_OUT_BUFFER_SIZE; ii++) {
					init_vdif_out(vpg_buf_cpu, ii);
				}
				df_num_insec = 0;
				ref_epoch = 0;
				secs_inre = 0;
				edh_psn = (uint64_t)0xffffffffffffffff;
				
				// initialize time-alignment  buffers
				for (jj=0; jj<VDIF_CHAN; jj++) {
					leftover_buffer[jj] = NULL;
					vdg_buf_cpu_time_aligned[jj] = (vdif_out_data_group_t *)malloc(sizeof(vdif_out_data_group_t));
					if (vdg_buf_cpu_time_aligned[jj] == NULL) {
						fprintf(stdout,"[TA] cannot allocate vdg_buf_cpu_time_aligned\n");
					} else {
						fprintf(stdout,"[TA] vdg_buf_cpu_time_aligned[%d] = %p\n",jj,vdg_buf_cpu_time_aligned[jj]);
					}
				}
				
				// and set our next state
				state = STATE_PROCESS;
			} // case STATE_IDLE
			
			case STATE_PROCESS: {

				while ((rv = hashpipe_databuf_wait_filled((hashpipe_databuf_t *)db_in, index_db_in)) != HASHPIPE_OK) {
					if (rv == HASHPIPE_TIMEOUT) { // index is not ready
						aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe input databuf timeout");
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
				
				aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "hashpipe databuf filled");
				
				fprintf(stdout,"%s:%s(%d): input buffer %d filled\n",__FILE__,__FUNCTION__,__LINE__,index_db_in);
				
				// grab the data at this index ...
				qs_buf.blocks[index_db_in] = (quantized_storage_t)db_in->blocks[index_db_in];
				// set local parameters in qs_buf.blocks, mainly local buffer
				qs_buf.blocks[index_db_in].vdg_buf_cpu = vdg_buf_cpu[qs_buf.blocks[index_db_in].gpu_id];
				// update timestamp information if needed
				skip_int32_t = 0;
				skip_frames = 0;
				num_usable_packets = qs_buf.blocks[index_db_in].N_32bit_words_per_chan / (VDIF_OUT_PKT_DATA_SIZE/4);
				if (!qs_buf.blocks[index_db_in].vdif_header_template.w0.invalid) {
					secs_inre = qs_buf.blocks[index_db_in].vdif_header_template.w0.secs_inre;
					df_num_insec = qs_buf.blocks[index_db_in].vdif_header_template.w1.df_num_insec;
					if (df_num_insec > (VDIF_OUT_FRAMES_PER_SECOND-num_usable_packets)) {
						skip_frames = VDIF_OUT_FRAMES_PER_SECOND-df_num_insec;
						fprintf(stdout,"%s:%s(%d): Skip %d x frames to move %d+%d to %d+%d\n",__FILE__,__FUNCTION__,__LINE__,skip_frames,secs_inre,df_num_insec,secs_inre+1,0);
						df_num_insec = 0;
						secs_inre++;
					}
					ref_epoch = qs_buf.blocks[index_db_in].vdif_header_template.w1.ref_epoch;
					uint64_t skip_picoseconds = qs_buf.blocks[index_db_in].vdif_header_template.edh_psn + SWARM_ZERO_DELAY_OFFSET_PICOSECONDS;
					int skip_samples = (int)floor((double)skip_picoseconds * R2DBE_RATE / 1e12);
					/* Skip samples with int32_t resolution, calculation
					 * is:
					 *   samples / bytes-per-sample / bytes-per-int32_t
					 */
					skip_int32_t = (int)floor((double)skip_samples / (8.0/(double)qs_buf.blocks[index_db_in].bit_depth) / (double)sizeof(int32_t));
					/* Number of frames that can be filled with the data
					 * left after skipping skip_int32_t worth of samples...
					 */
					num_usable_packets = (qs_buf.blocks[index_db_in].N_32bit_words_per_chan - skip_int32_t) / (VDIF_OUT_PKT_DATA_SIZE/4);
					/* ...and after skipping skip_frames worth of frames
					 */
					num_usable_packets = num_usable_packets - skip_frames;
					/* This is the number of int32_t which needs to be
					 * carried over to the next pass per channel.
					 */
					leftover_int32_t = qs_buf.blocks[index_db_in].N_32bit_words_per_chan - skip_int32_t - num_usable_packets*(VDIF_OUT_PKT_DATA_SIZE/4) - skip_frames*(VDIF_OUT_PKT_DATA_SIZE/4);
					fprintf(stdout,"%s:%s(%d): Skip %d x int32_t (%d samples, %lu ps), carry over %d x int32_t\n",__FILE__,__FUNCTION__,__LINE__,skip_int32_t,skip_samples,skip_picoseconds,leftover_int32_t);
					edh_psn = 0x01;
					fprintf(stdout,"%s:%s(%d): initial timestamp is %u@%u.%u (psn = %lu)\n",__FILE__,__FUNCTION__,__LINE__,ref_epoch,secs_inre,df_num_insec,edh_psn);
				}
				
				print_quantized_storage(&qs_buf.blocks[index_db_in], "");
				
				// start asynchronous data transfer from GPU to host
				transfer_vdif_group_to_cpu(&qs_buf, index_db_in);
				// this call blocks until transfer is done
				check_transfer_vdif_group_to_cpu_complete(&qs_buf, index_db_in);
				// let hashpipe know we're done with the buffer (for now)
				hashpipe_databuf_set_free((hashpipe_databuf_t *)db_in, index_db_in);
				
				fprintf(stdout,"%s:%s(%d): input buffer %d free\n",__FILE__,__FUNCTION__,__LINE__,index_db_in);
				
				// copy data to time-aligned vdg buffer
				for (jj=0; jj<VDIF_CHAN; jj++) {
					/* Destination is usually the start of the time-
					 * aligned vdg buffer.
					 */
					void *dst = (void *)(vdg_buf_cpu_time_aligned[jj]->chan[jj].datas);
					/* Source is the newly received vdg data offset by
					 * skip_int32_t x sizeof(int32_t). Note that
					 * skip_int32_t is updated with each iteration, and
					 * always zero after first iteration.
					 */
					void *src = (void *)vdg_buf_cpu[index_db_in]->chan[jj].datas + sizeof(int32_t)*skip_int32_t + VDIF_OUT_PKT_DATA_SIZE*skip_frames;
					/* If there is data left over from previous pass,
					 * copy that into the time-aligned vdg buffer.
					 */
					if (leftover_buffer[jj] != NULL) {
						memcpy(dst,(void *)leftover_buffer[jj],leftover_int32_t*sizeof(int32_t));
						/* Offset destination to account for leftovers
						 * from previous iteration already copied in.
						 */
						dst += leftover_int32_t*sizeof(int32_t);
					}
					/* Now copy newly received data, the copy size is
					 * total received int32_t minus skipped (discarded
					 * at start of input) minus leftover (copied to
					 * buffer for use in next iteration).
					 */
					memcpy(dst,src,(qs_buf.blocks[index_db_in].N_32bit_words_per_chan - skip_int32_t - leftover_int32_t)*sizeof(int32_t));
					/* Copy leftover data into buffer */
					if (leftover_int32_t > 0) {
						// allocate memory if none yet
						if (leftover_buffer[jj] == NULL) {
							leftover_buffer[jj] = (int32_t *)malloc(leftover_int32_t*sizeof(int32_t));
						}
						/* Source for the copy is leftover number of
						 * int32_t less than the total number of int32_t
						 * at the input.
						 */
						src = (void *)vdg_buf_cpu[index_db_in]->chan[jj].datas + (qs_buf.blocks[index_db_in].N_32bit_words_per_chan - leftover_int32_t)*sizeof(int32_t);
						memcpy((void *)leftover_buffer[jj],src,leftover_int32_t*sizeof(int32_t));
					}
				}
				
				fprintf(stdout,"%s:%s(%d): [TA]: %d / %d usable packets\n",__FILE__,__FUNCTION__,__LINE__,num_usable_packets,qs_buf.blocks[index_db_in].N_32bit_words_per_chan / (VDIF_OUT_PKT_DATA_SIZE/4));
				// copy VDIF data to VDIF packet memory, update headers
				for (ii=0; ii<num_usable_packets; ii++) {
					for (jj=0; jj<VDIF_CHAN; jj++) {
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].data = vdg_buf_cpu_time_aligned[jj]->chan[jj].datas[ii];
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w0.secs_inre = secs_inre;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w1.ref_epoch = ref_epoch;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w1.df_num_insec = df_num_insec;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w3.stationID = meta_station;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w3.threadID = 0;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w3.bps = qs_buf.blocks[index_db_in].bit_depth - 1; // VDIF bits-per-sample adds 1
						// mark polarization / bdc sideband / receiver sideband
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w4.pol = jj;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w4.bdc_sideband = meta_bdc; // <--- Apr2017, Quad2; Apr2017, Quad3 --> vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w4.bdc_sideband = 1;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.w4.rx_sideband = meta_rx;
						vpg_buf_cpu[index_db_in]->chan[jj].packets[ii].header.edh_psn = edh_psn;
					}
					df_num_insec++;
					if (df_num_insec == VDIF_OUT_FRAMES_PER_SECOND) {
						df_num_insec = 0;
						secs_inre++;
					}
					edh_psn++;
					// TODO: check for psn wrapping?
				}
				current_time = time(NULL);
				fprintf(stdout,"%s:%s(%d): VDIF time is %u.%u ; current Epoch time is %d\n",__FILE__,__FUNCTION__,__LINE__,secs_inre,df_num_insec,(int)current_time);
				
				// TODO: send data over network to sgrx
				//~ tx_tries = MAX_TX_TRIES;
				//~ do {
					// TODO: check number of frames received at other end
					// against the total number of frames to be sent.
				for (ii=0; ii<VDIF_CHAN; ii++) {
					frames_sent[ii] = tx_frames(sockfd[ii], (void *)&(vpg_buf_cpu[index_db_in]->chan[ii]), num_usable_packets, sizeof(vdif_out_packet_t));
					fprintf(stdout,"%s:%s(%d): tx_frames on sockfd[%d] returned %d\n",__FILE__,__FUNCTION__,__LINE__,ii,frames_sent[ii]);
					if (frames_sent[ii] < 0) {
						frames_sent[ii] = 0;
						aphids_set(&aphids_ctx,"status:net","error-on-send");
						hashpipe_error(__FUNCTION__,"error sending data");
						state = STATE_ERROR;
						break; // for ii over VDIF_CHAN
					}
				}
				if (state == STATE_ERROR) {
					break; // switch(state)
				}
				//~ } while (tx_tries-- > 0);
				//~ if (tx_tries == 0) {
					//~ 
				//~ }
				
				// update the index modulo the maximum buffer depth
				index_db_in = (index_db_in + 1) % db_in->header.n_block;
				
				// update aphids statistics
				aphids_update(&aphids_ctx);
				break;
			} // case STATE_PROCESS
		}// switch(state)
	} // end while(run_threads())
	
	aphids_set(&aphids_ctx,"status:net","ending transmission");
	for (ii=0; ii<VDIF_CHAN; ii++) {
		if (tx_frames(sockfd[ii], (void *)&(vpg_buf_cpu[index_db_in]->chan[ii]), 0, sizeof(vdif_out_packet_t)) != 0) {
			hashpipe_error(__FUNCTION__,"%s:%s(%d):Could not send end-of-transmission to sockfd[%d]",__FILE__,__FUNCTION__,__LINE__,ii);
		}
	}
	
	aphids_set(&aphids_ctx,"status:net","end-of-transmission sent");
	// close connection to server
	for (ii=0; ii<VDIF_CHAN; ii++) {
		if (sockfd >= 0) {
			close(sockfd[ii]);
		}
	}
	
	aphids_set(&aphids_ctx,"status:net","connection closed");
	// destroy aphids context and exit
	aphids_destroy(&aphids_ctx);
	
	// clean up time-alignment buffers
	for (jj=0; jj<VDIF_CHAN; jj++){
		if (leftover_buffer[jj] != NULL) {
			free(leftover_buffer[jj]);
		}
		if (vdg_buf_cpu_time_aligned[jj] != NULL) {
			free(vdg_buf_cpu_time_aligned[jj]);
		}
	}
	
	return NULL;
}

static hashpipe_thread_desc_t vdif_out_net_thread = {
	name: "vdif_out_net_thread",
	skey: "VDIFOUT",
	init: NULL,
	run:  run_method,
	ibuf_desc: {vdif_out_databuf_create},
	obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor() {
	register_hashpipe_thread(&vdif_out_net_thread);
}
