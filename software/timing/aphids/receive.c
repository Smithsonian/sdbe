/*
 * Recieves VDIF frames and does nothing
 *
 * Follows methods and conventions set in vdif_in_net_thread.c
 * 
 * @author: Katherine Rosenfeld
 * @date: 10/2015
 * */

#include <stdio.h>
#include <syslog.h>
//#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#include "sgcomm_net.h"

// network parameters for hamster >> Mark6-4016
#define RX_HOST "192.168.10.63"
#define RX_PORT ((uint16_t)54323)

int main(int argc, char *argv[]){

	int rv = -1;
	int i;

	// for sgcomm_net.h interface
	int sockfd_server = 0;
	int sockfd_data = 0;
	ssize_t size = 0;
	int timeouts = 5;

	// for VDIF handling 
	void *received_vdif_packets = NULL;
	ssize_t n_received_vdif_packets = 0;
	ssize_t index_received_vdif_packets = 0;
	ssize_t N_ALL_VDIF_PACKETS = 0;

	// create socket and bind
	sockfd_server = make_socket_bind_listen(RX_HOST, RX_PORT);

	// accept an incoming connection	
	do {
		sockfd_data = accept_connection(sockfd_server);
		if (sockfd_data < 0) {
			if (sockfd_data == ERR_NET_TIMEOUT) {
				fprintf(stderr,"%s:%d:timeout-on-connect (%d)\n",__FILE__,__LINE__,timeouts);
			}
		} else {
			fprintf(stderr,"%s:%d:error-on-connect\n",__FILE__,__LINE__);
		}
	} while (--timeouts > 0);

	do {
		// read batch of VDIF packets into local buffer
		//   rv is the expected number of frames
		//   n_received_vdif_packets is the actual number of frames received
		//   size is the size of frames received
		rv = rx_frames(sockfd_data, &received_vdif_packets, &n_received_vdif_packets, &size);

		fprintf(stdout,"%s:%d: received %d packets\n",__FILE__,__LINE__,(int)n_received_vdif_packets);
		if (rv < 0){
			fprintf(stderr,"%s:%d:error-on-receive\n",__FILE__,__LINE__);
			break;
		}
		N_ALL_VDIF_PACKETS += n_received_vdif_packets;

		// do nothing and free buffer
		//for (i=0; i< n_received_vdif_packets; ++i){
	//		free(received_vdif_packets[i]);
		//}
		free(received_vdif_packets);
	} while (rv != 0);

	fprintf(stdout,"%s:%d: VDIF done, received %ld packets\n",__FILE__,__LINE__,(long int)N_ALL_VDIF_PACKETS);

	// close sockets
	close(sockfd_server);
	close(sockfd_data);

	fprintf(stdout,"%s:%d: closed shop\n",__FILE__,__LINE__);

	return 1;
}
