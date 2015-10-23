/*
 * Sends fake VDIF frames 
 *
 * Follows methods and conventions set in vdif_in_net_thread.c
 * 
 * B-engine data are grouped in packets which are grouped to send over
 * network in frames.
 *
 * @author: Katherine Rosenfeld
 * @date: 10/2015
 * */

#include <stdio.h>
#include <syslog.h>
//#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#include "vdif_in_databuf.h"
#include "sample_rates.h"

#include "sgcomm_net.h"

// network parameters for Mark6-4015 >> hamster
#define TX_HOST "192.168.10.10"
#define TX_PORT ((uint16_t)12345)

int64_t get_packet_b_count(vdif_in_header_t *vdif_pkt_hdr) {
        int64_t b = 0;
        b |= ((int64_t)(vdif_pkt_hdr->beng.b_upper)&(int64_t)0x00000000FFFFFFFF) << 8;
        b |= (int64_t)(vdif_pkt_hdr->beng.b_lower)&(int64_t)0x00000000000000FF;
        return b;
}

void set_packet_b_count(vdif_in_header_t *vdif_pkt_hdr, int64_t b){
	vdif_pkt_hdr->beng.b_upper = (uint32_t) ((b&0xFFFFFFFF00) >> 8) ;
	vdif_pkt_hdr->beng.b_lower = (uint8_t) (b&0xFF);
}

int main(int argc, char *argv[]){

	int i,j;
	int blocks_to_send = 0;
	int beng_frames_per_block = 1;
	int rv = 0;
	int total_tx_frames_sent = 0;
	vdif_in_packet_t *beng_buf;

	// for sgcomm_net.h interface
	int sockfd = 0;
	int tx_frames_sent = 0;

	// transmitter parameters
	if (argc > 1)
		blocks_to_send = atoi(argv[1]);
	if (argc > 2)
		beng_frames_per_block = atoi(argv[2]);	

	fprintf(stdout,"%s:%d:transmitting %d B-ENG frames\n",__FILE__,__LINE__,blocks_to_send*beng_frames_per_block);

	// allocate beng_buf in bytes 
	int64_t b_counter = 0;
	fprintf(stdout,"%s:%d:size of beng_buf %d bytes\n",__FILE__,__LINE__,beng_frames_per_block*VDIF_PER_BENG_FRAME*(VDIF_IN_PKT_DATA_SIZE+VDIF_IN_PKT_HEADER_SIZE));
	//beng_buf = (uint32_t *) malloc(beng_frames_per_block*VDIF_PER_BENG_FRAME*(VDIF_IN_PKT_DATA_SIZE+VDIF_IN_PKT_HEADER_SIZE));
	beng_buf = (vdif_in_packet_t *) malloc((size_t) beng_frames_per_block*VDIF_PER_BENG_FRAME*sizeof(vdif_in_packet_t));
	for (i=0; i<beng_frames_per_block; ++i){
		for (j=0;j<VDIF_PER_BENG_FRAME; ++j){
			//set_packet_b_count(&((beng_buf+i*VDIF_PER_BENG_FRAME+j)->header), b_counter);
			set_packet_b_count(&(beng_buf[i*VDIF_PER_BENG_FRAME+j].header), b_counter);
		}
		++b_counter;
	}

	// setup network connection
	sockfd = make_socket_connect(TX_HOST,TX_PORT);
	fprintf(stdout,"%s:%d: socked opened with descriptior %d\n",__FILE__,__LINE__,sockfd);
	if (sockfd < 0){
		fprintf(stderr,"%s:%d:error-on-connect\n",__FILE__,__LINE__);
	}

	// loop through blocks
	do {
		// send frames
		//   APHIDS assumes that 1 tx_frame = 1 vdif packet
		tx_frames_sent= tx_frames(sockfd, (void *) beng_buf, beng_frames_per_block*VDIF_PER_BENG_FRAME,sizeof(vdif_in_packet_t));
		total_tx_frames_sent += tx_frames_sent;
		fprintf(stdout,"%s:%d:tx_frames on sockfd returned %d\n",__FILE__,__LINE__,tx_frames_sent);
		// increase counter
	} while (total_tx_frames_sent < blocks_to_send*beng_frames_per_block*VDIF_PER_BENG_FRAME);

	// send end-of-transmission 
	if (tx_frames(sockfd, (void *) beng_buf, 0, sizeof(vdif_in_packet_t)) != 0) {
		fprintf(stderr,"%s:%d:error on sending end-of-transmission\n",__FILE__,__LINE__);
	}

	// close connection to server 
	close(sockfd);

	// free B-eng buffer
	free(beng_buf);

	return 1;
}
