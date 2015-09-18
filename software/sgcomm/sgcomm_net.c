#include <errno.h>
#include <netdb.h> 
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/socket.h>

#include "sgcomm_net.h"
//~**~#include "sgcomm_report.h"

/* Initialize socket address, taken from some example in GNU C Library
 *   reference.
 * */
int init_sockaddr(struct sockaddr_in *name,const char *hostname,uint16_t port);

/* Receive a single frame over a socket.
 * Arguments:
 *   int sockfd -- Socket to be read from
 *   void *buf -- Pointer to memory where received frame should be stored
 *   ssize_t buflen -- Number of bytes that will be occupied by frame
 * Return:
 *   int -- 0 on successful receiption, or < 0 for error (see enumerated
 *     error codes in sgcomm_net.h)
 * Notes:
 *   This method checks for timeout on the socket. See parameters 
 *     defined in sgcomm_net.h 
 * */
int rx_frame(int sockfd, void *buf, ssize_t buflen);

/* Transmit a single frame over a socket.
 * Arguments:
 *   int sockfd -- Socket to be written to
 *   void *buf -- Pointer to address where frame is stored
 *   ssize_t buflen -- Number of bytes occupied by frame
 * Return:
 *   int -- 0 on successful transmission, or < 0 for error (see 
 *     enumerated error codes in sgcomm_net.h)
 * Notes:
 *   This method checks for timeout on the socket. See parameters 
 *     defined in sgcomm_net.h 
 * */
int tx_frame(int sockfd, void *buf, ssize_t buflen);

int init_sockaddr(struct sockaddr_in *name,const char *hostname,uint16_t port) {
	struct hostent *hostinfo;
	
	name->sin_family = AF_INET;
	name->sin_port = htons (port);
	hostinfo = gethostbyname (hostname);
	if (hostinfo == NULL) {
		fprintf(stderr,"%s:%s(%d)Unknown host %s.\n", hostname);
		return ERR_NET_UNKNOWN_HOST;
	}
	name->sin_addr = *(struct in_addr *) hostinfo->h_addr;
	return 0;
}

int make_socket_connect(const char *host, uint16_t port) {
	int sockfd;
	struct sockaddr_in serv_addr;
	
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) {
		fprintf(stderr,"%s:%s(%d):Unable to open socket",__FILE__,__FUNCTION__,__LINE__);
		perror("socket()");
		return ERR_NET_CANNOT_OPEN_SOCKET;
	}
//~NOTIMEOUT~	/* Set a timeout on the data socket */
//~NOTIMEOUT~	struct timeval timeout;
//~NOTIMEOUT~	timeout.tv_sec = TIMEOUT_SEC;
//~NOTIMEOUT~	timeout.tv_usec = TIMEOUT_USEC;
//~NOTIMEOUT~	if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO,(void *)&timeout,sizeof(timeout)) < 0) {
//~NOTIMEOUT~		//~**~log_message(RL_WARNING,"%s:%s(%d):Unable to set timeout on receiving socket",__FILE__,__FUNCTION__,__LINE__);
//~NOTIMEOUT~	}
	if (init_sockaddr(&serv_addr, host, port) != 0) {
		fprintf(stderr,"%s:%s(%d):Unable to resolve hostname",__FILE__,__FUNCTION__,__LINE__);
		close(sockfd);
		return ERR_NET_UNKNOWN_HOST;
	}
	if (connect(sockfd,(struct sockaddr *)&serv_addr,sizeof(serv_addr)) < 0) {
		fprintf(stderr,"%s:%s(%d):Unable to connect socket",__FILE__,__FUNCTION__,__LINE__);
		perror("connect");
		close(sockfd);
		return ERR_NET_CANNOT_CONNECT_SOCKET;
	}
	return sockfd;
}

int make_socket_bind_listen(const char *host, uint16_t port) {
	int sockfd;
	struct sockaddr_in serv_addr;
	
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) {
		fprintf(stderr,"%s:%s(%d):Unable to open socket",__FILE__,__FUNCTION__,__LINE__);
		perror("socket");
		return ERR_NET_CANNOT_OPEN_SOCKET;
	}
//~NOTIMEOUT~	/* Set a timeout on the listen socket */
//~NOTIMEOUT~	struct timeval timeout;
//~NOTIMEOUT~	timeout.tv_sec = TIMEOUT_SEC;
//~NOTIMEOUT~	timeout.tv_usec = TIMEOUT_USEC;
//~NOTIMEOUT~	if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO,(void *)&timeout,sizeof(timeout)) < 0) {
//~NOTIMEOUT~		//~**~log_message(RL_WARNING,"%s:%s(%d):Unable to set timeout on receiving socket",__FILE__,__FUNCTION__,__LINE__);
//~NOTIMEOUT~	}
	int val = 1;
	if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(int)) < 0) {
		// log error, but try to continue anyway
		perror("setsockopt");
	}
	init_sockaddr(&serv_addr, host, port);
	if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
		fprintf(stderr,"%s:%s(%d):Unable to bind socket",__FILE__,__FUNCTION__,__LINE__);
		perror("bind");
		close(sockfd);
		return ERR_NET_CANNOT_BIND_SOCKET;
	}
	if (listen(sockfd,1) < 0) {
		fprintf(stderr,"%s:%s(%d):Unable to listen on socket",__FILE__,__FUNCTION__,__LINE__);
		perror("listen");
		close(sockfd);
		return ERR_NET_CANNOT_LISTEN_ON_SOCKET;
	}
	return sockfd;
}

int accept_connection(int sockfd) {
	int newsockfd;
	struct sockaddr_in client_addr;
	socklen_t client_len;
	int timeouts;
	
	client_len = sizeof(client_addr);
	timeouts = TIMEOUT_COUNT;
	do {
		newsockfd = accept(sockfd, (struct sockaddr *)&client_addr, &client_len);
		if (newsockfd < 0) {
			if (errno == EAGAIN) {
				if (timeouts-- <= 0) {
					return ERR_NET_TIMEOUT;
				} else {
				}
			} else {
				//~**~log_message(RL_ERROR,"%s:%s(%d):Unable to accept connection",__FILE__,__FUNCTION__,__LINE__);
				return ERR_NET_CANNOT_ACCEPT_CONNECTION;
			}
		} else {
			break;
		}
	} while (1);
	
//~NOTIMEOUT~	/* Set a timeout on the data socket */
//~NOTIMEOUT~	struct timeval timeout;
//~NOTIMEOUT~	timeout.tv_sec = TIMEOUT_SEC;
//~NOTIMEOUT~	timeout.tv_usec = TIMEOUT_USEC;
//~NOTIMEOUT~	if (setsockopt(newsockfd, SOL_SOCKET, SO_RCVTIMEO,(void *)&timeout,sizeof(timeout)) < 0) {
//~NOTIMEOUT~		//~**~log_message(RL_WARNING,"%s:%s(%d):Unable to set timeout on receiving socket",__FILE__,__FUNCTION__,__LINE__);
//~NOTIMEOUT~	}
	return newsockfd;
}

int rx_frame(int sockfd, void *buf, ssize_t buflen) {
	ssize_t bytes_received, n;
	int timeouts;
	
	bytes_received = 0;
	timeouts = TIMEOUT_COUNT;
	do {
		//~ //~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Point of read",__FILE__,__FUNCTION__,__LINE__);
		n = read(sockfd,buf+bytes_received,buflen-bytes_received);
		if (n < 0) {
			if (errno == EAGAIN) {
				if (timeouts-- <= 0) {
					fprintf(stdout,"%s:%s(%d):WARNING Receiving frame timed out, quiting with code %d (%lu bytes received and discarded)\n",__FILE__,__FUNCTION__,__LINE__,ERR_NET_TIMEOUT,bytes_received);
					return ERR_NET_TIMEOUT;
				} else {
					fprintf(stdout,"%s:%s(%d):WARNING Timeout on receiving frame, %d retries left\n",__FILE__,__FUNCTION__,__LINE__,timeouts);
				}
			} else {
				//~**~log_message(RL_ERROR,"%s:%s(%d):Unable to read from socket, return code %d",__FILE__,__FUNCTION__,__LINE__,ERR_NET_CANNOT_READ_SOCKET);
				fprintf(stderr,"%s:%s(%d):Unexpected error on receiving frame (%d error)\n",__FILE__,__FUNCTION__,__LINE__,errno);
				perror("rx_frame");
				return ERR_NET_CANNOT_READ_SOCKET;
			}
		} else if (n == 0) {
			//~**~log_message(RL_NOTICE,"%s:%s(%d):Socket apparently closed, return zero %d after %lu bytes",__FILE__,__FUNCTION__,__LINE__,n,bytes_received);
			return 0;
		} else {
			bytes_received += n;
		}
	} while (bytes_received < buflen);
	//~ //~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Received %lu bytes",__FILE__,__FUNCTION__,__LINE__,bytes_received);
	return 0;
}

int tx_frame(int sockfd, void *buf, ssize_t buflen) {
	ssize_t bytes_transmitted, n;
	int timeouts;
	
	bytes_transmitted = 0;
	timeouts = TIMEOUT_COUNT;
	do {
		//~ //~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Point of write",__FILE__,__FUNCTION__,__LINE__);
		n = write(sockfd,buf+bytes_transmitted,buflen-bytes_transmitted);
		if (n < 0) {
			if (errno == EAGAIN) {
				if (timeouts-- <= 0) {
					//~**~log_message(RL_WARNING,"%s:%s(%d):Sending frame timed out, quiting with code %d (%lu bytes sent).",__FILE__,__FUNCTION__,__LINE__,ERR_NET_TIMEOUT,bytes_transmitted);
					return ERR_NET_TIMEOUT;
				} else {
					fprintf(stdout,"%s:%s(%d):WARNING Timeout on sending frame, %d retries left.",__FILE__,__FUNCTION__,__LINE__,timeouts);
				}
			} else {
				//~**~log_message(RL_ERROR,"%s:%s(%d):Unable to write to socket, return code %d",__FILE__,__FUNCTION__,__LINE__,ERR_NET_CANNOT_WRITE_SOCKET);
				fprintf(stderr,"%s:%s(%d):Unexpected error on sending frame (%d error)\n",__FILE__,__FUNCTION__,__LINE__,errno);
				perror("tx_frame");
				return ERR_NET_CANNOT_WRITE_SOCKET;
			}
		} else if (n == 0) {
			//~**~log_message(RL_NOTICE,"%s:%s(%d):Socket apparently closed, return zero %d after %lu bytes sent",__FILE__,__FUNCTION__,__LINE__,n,bytes_transmitted);
			return 0;
		} else {
			bytes_transmitted += n;
		}
	} while (bytes_transmitted < buflen);
	//~ //~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Sent %lu bytes",__FILE__,__FUNCTION__,__LINE__,bytes_transmitted);
	return 0;
}

int rx_frames(int sockfd, void **buf, ssize_t *nmem, ssize_t *size) {
	int err_code = 0;
	ssize_t frames_received = 0;
	handshake tx;
	handshake rx = {
		.frame_size = 0,
		.n_frames = 0
	};
	/* Wait for TX to initiate handshake */
	//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting for handshake",__FILE__,__FUNCTION__,__LINE__);
	err_code = rx_frame(sockfd, (void *)&rx, sizeof(handshake));
	if (err_code < 0) {
		//~**~fprintf(stderr,"%s:%s(%d):Communication failure on receiving handshake",__FILE__,__FUNCTION__,__LINE__);
		return err_code;
	}
	/* Complete handshake */
	//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Sending first response",__FILE__,__FUNCTION__,__LINE__);
	// On first response ...
	tx.frame_size = rx.frame_size; // ... acknowledge frame size ...
	tx.n_frames = 0; // ... but set number of frames to zero
	err_code = tx_frame(sockfd, (void *)&tx, sizeof(handshake));
	if (err_code < 0) {
		fprintf(stderr,"%s:%s(%d):Communication failure on completing handshake\n",__FILE__,__FUNCTION__,__LINE__);
		return ERR_NET_COMMS_PROTOCOL_FAILURE;
	}
	/* Receive data */
	if (rx.n_frames > 0) {
		//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Receiving data",__FILE__,__FUNCTION__,__LINE__);
		/* Create buffer to store expected data and set size parameters */
		*buf = malloc(rx.n_frames*rx.frame_size);
		for (frames_received=0; frames_received<rx.n_frames; frames_received++) {
			err_code = rx_frame(sockfd, *buf+frames_received*rx.frame_size, rx.frame_size);
			if (err_code < 0) {
				fprintf(stderr,"%s:%s(%d):Communication failure during data receiving\n",__FILE__,__FUNCTION__,__LINE__);
				return ERR_NET_COMMS_PROTOCOL_FAILURE;
				//~ break;
			}
		}
	}
	*size = rx.frame_size; // set frame size ...
	*nmem = frames_received; // ... and number of frames in buffer
	/* Conclude communication by sending last response */
	//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Sending last response",__FILE__,__FUNCTION__,__LINE__);
	// On last response ...
	tx.frame_size = rx.frame_size; // ... acknowledge frame size ...
	tx.n_frames = frames_received; // ... but set number of frames equal to total received
	err_code = tx_frame(sockfd, (void *)&tx, sizeof(handshake));
	if (err_code < 0) {
		fprintf(stderr,"%s:%s(%d):Communication failure on sending last response\n",__FILE__,__FUNCTION__,__LINE__);
		return err_code;
	}
	//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Successful communication, received %lu frames",__FILE__,__FUNCTION__,__LINE__,*nmem);
	return rx.n_frames;
}

int tx_frames(int sockfd, void *buf, ssize_t nmem, ssize_t size) {
	int err_code = 0;
	ssize_t frames_transmitted = 0;
	handshake tx = {
		.frame_size = size,
		.n_frames = nmem
	};
	handshake rx = {
		.frame_size = 0,
		.n_frames = nmem
	};
	/* Send handshake to initialize communication */
	//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Initiating handshake",__FILE__,__FUNCTION__,__LINE__);
	err_code = tx_frame(sockfd,(void *)&tx,sizeof(handshake));
	if (err_code < 0) {
		//~**~log_message(RL_ERROR,"%s:%s(%d):Communication failure on initiating handshake",__FILE__,__FUNCTION__,__LINE__);
		return err_code;
	}
	/* Wait for RX ready */
	//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting on first response",__FILE__,__FUNCTION__,__LINE__);
	err_code = rx_frame(sockfd,(void *)&rx,sizeof(handshake));
	if (err_code < 0) {
		fprintf(stderr,"%s:%s(%d):Communication failure waiting on first response (%d error)\n",__FILE__,__FUNCTION__,__LINE__,errno);
		return ERR_NET_COMMS_PROTOCOL_FAILURE;
	}
	/* Transmit data */
	if (rx.frame_size == tx.frame_size && rx.n_frames == 0) {
		//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Transmitting data",__FILE__,__FUNCTION__,__LINE__);
		for (frames_transmitted=0; frames_transmitted<nmem; frames_transmitted++) {
			err_code = tx_frame(sockfd,buf+frames_transmitted*size, size);
			if (err_code < 0) {
				fprintf(stderr,"%s:%s(%d):Communication failure during data send\n",__FILE__,__FUNCTION__,__LINE__);
				return ERR_NET_COMMS_PROTOCOL_FAILURE;
				//~ break;
			}
		}
	} else {
		fprintf(stderr,"%s:%s(%d):Invalid ACK received: frame_size = %u (expected %u), n_frames = %u (expected %u)\n",__FILE__,__FUNCTION__,__LINE__,rx.frame_size,size,rx.n_frames,0);
		return ERR_NET_COMMS_PROTOCOL_FAILURE;
	}
	/* Wait for handshake to end communcation */
	//~**~log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting on last response",__FILE__,__FUNCTION__,__LINE__);
	rx.frame_size = 0;
	rx.n_frames = 0;
	err_code = rx_frame(sockfd,(void *)&rx, sizeof(handshake));
	if (err_code < 0) {
		fprintf(stderr,"%s:%s(%d):Communication failure on last response\n",__FILE__,__FUNCTION__,__LINE__);
		return ERR_NET_COMMS_PROTOCOL_FAILURE;
	}
	if (rx.frame_size == tx.frame_size) {
		if (rx.n_frames < nmem) {
			//~**~log_message(RL_WARNING,"%s:%s(%d):Transmitted %lu / %lu packets, but only %lu received at other end",__FILE__,__FUNCTION__,__LINE__,frames_transmitted,nmem,rx.n_frames);
		}
		return rx.n_frames;
	} else {
		fprintf(stderr,"%s:%s(%d):Invalid ACK received: frame_size = %u (expected %u), n_frames = %u (expected > 0)\n",__FILE__,__FUNCTION__,__LINE__,rx.frame_size,size,rx.n_frames);
		return ERR_NET_COMMS_PROTOCOL_FAILURE;
	}
}
