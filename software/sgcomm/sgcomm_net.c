#include <errno.h>
#include <netdb.h> 
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/socket.h>

#include "sgcomm_net.h"
#include "sgcomm_report.h"

int init_sockaddr(struct sockaddr_in *name,const char *hostname,uint16_t port) {
	struct hostent *hostinfo;
	
	name->sin_family = AF_INET;
	name->sin_port = htons (port);
	hostinfo = gethostbyname (hostname);
	if (hostinfo == NULL) {
		log_message(RL_ERROR,"%s:%s(%d)Unknown host %s.\n", hostname);
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
		log_message(RL_ERROR,"%s:%s(%d):Unable to open socket",__FILE__,__FUNCTION__,__LINE__);
		return ERR_NET_CANNOT_OPEN_SOCKET;
	}
	if (init_sockaddr(&serv_addr, host, port) != 0) {
		log_message(RL_ERROR,"%s:%s(%d):Unable to resolve hostname",__FILE__,__FUNCTION__,__LINE__);
		close(sockfd);
		return ERR_NET_UNKNOWN_HOST;
	}
	if (connect(sockfd,(struct sockaddr *)&serv_addr,sizeof(serv_addr)) < 0) {
		log_message(RL_ERROR,"%s:%s(%d):Unable to connect socket",__FILE__,__FUNCTION__,__LINE__);
		perror("");
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
		log_message(RL_ERROR,"%s:%s(%d):Unable to open socket",__FILE__,__FUNCTION__,__LINE__);
		return ERR_NET_CANNOT_OPEN_SOCKET;
	}
	init_sockaddr(&serv_addr, host, port);
	if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
		log_message(RL_ERROR,"%s:%s(%d):Unable to bind socket",__FILE__,__FUNCTION__,__LINE__);
		close(sockfd);
		return ERR_NET_CANNOT_BIND_SOCKET;
	}
	if (listen(sockfd,1) < 0) {
		log_message(RL_ERROR,"%s:%s(%d):Unable to listen on socket",__FILE__,__FUNCTION__,__LINE__);
		close(sockfd);
		return ERR_NET_CANNOT_LISTEN_ON_SOCKET;
	}
	return sockfd;
}

int accept_connection(int sockfd) {
	int newsockfd;
	struct sockaddr_in client_addr;
	int client_len;
	
	client_len = sizeof(client_addr);
	newsockfd = accept(sockfd, (struct sockaddr *)&client_addr, &client_len);
	if (newsockfd < 0) {
		log_message(RL_ERROR,"%s:%s(%d):Unable to accept connection",__FILE__,__FUNCTION__,__LINE__);
		perror("accept_connection");
		return ERR_NET_CANNOT_ACCEPT_CONNECTION;
	}
	
	/* Set a timeout on the data socket */
	struct timeval timeout;
	timeout.tv_sec = TIMEOUT_SEC;
	timeout.tv_usec = TIMEOUT_USEC;
	if (setsockopt(newsockfd, SOL_SOCKET, SO_RCVTIMEO,(void *)&timeout,sizeof(timeout)) < 0)
		log_message(RL_WARNING,"%s:%s(%d):Unable to set timeout on receiving socket",__FILE__,__FUNCTION__,__LINE__);
	return newsockfd;
}

int rx_frame(int sockfd, void *buf, ssize_t buflen) {
	ssize_t bytes_received, n;
	int timeouts;
	
	bytes_received = 0;
	timeouts = 3;
	do {
		//~ //~ log_message(RL_DEBUGVVV,"%s:%s(%d):Point of read",__FILE__,__FUNCTION__,__LINE__);
		n = read(sockfd,buf+bytes_received,buflen-bytes_received);
		if (n < 0) {
			if (errno == EAGAIN) {
				if (timeouts-- <= 0) {
					log_message(RL_WARNING,"%s:%s(%d):Receiving frame timed out, quiting with code %d (%lu bytes received and discarded).",__FILE__,__FUNCTION__,__LINE__,ERR_NET_READ_TIMEOUT,bytes_received);
					return ERR_NET_READ_TIMEOUT;
				} else
					log_message(RL_WARNING,"%s:%s(%d):Timeout on receiving frame, %d retries left.",__FILE__,__FUNCTION__,__LINE__,timeouts);
			} else {
				log_message(RL_ERROR,"%s:%s(%d):Unable to read from socket, return code %d",__FILE__,__FUNCTION__,__LINE__,ERR_NET_CANNOT_READ_SOCKET);
				return ERR_NET_CANNOT_READ_SOCKET;
			}
		} else if (n == 0) {
			log_message(RL_NOTICE,"%s:%s(%d):Socket apparently closed, return zero %d",__FILE__,__FUNCTION__,__LINE__,n);
				return 0;
		} else {
			bytes_received += n;
			//~ break;
		}
		//~ if (errno == EAGAIN) {
			//~ printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		//~ }
	} while (bytes_received < buflen);
	//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Received %lu bytes",__FILE__,__FUNCTION__,__LINE__,bytes_received);
	return bytes_received;
}

int tx_frame(int sockfd, void *buf, ssize_t buflen) {
	ssize_t bytes_transmitted, n;
	
	bytes_transmitted = 0;
	do {
		n = write(sockfd,buf+bytes_transmitted,buflen-bytes_transmitted);
		if (n < 0) {
			log_message(RL_ERROR,"%s:%s(%d):Unable to write to socket",__FILE__,__FUNCTION__,__LINE__);
			return ERR_NET_CANNOT_WRITE_SOCKET;
		}
		bytes_transmitted += n;
	} while (bytes_transmitted < buflen);
	return bytes_transmitted;
}

int rx_frames(int sockfd, void **buf, ssize_t *nmem, ssize_t *size) {
	int num_bytes_or_err = 0;
	ssize_t frames_received = 0;
	handshake tx;
	handshake rx = {
		.frame_size = 0,
		.n_frames = 0
	};
	/* Wait for TX to initiate handshake */
	log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting for handshake",__FILE__,__FUNCTION__,__LINE__);
	num_bytes_or_err = rx_frame(sockfd, (void *)&rx, sizeof(handshake));
	if (num_bytes_or_err <= 0) {
		log_message(RL_ERROR,"%s:%s(%d):Communication failure on receiving handshake",__FILE__,__FUNCTION__,__LINE__);
		return num_bytes_or_err;
	}
	/* Complete handshake */
	log_message(RL_DEBUGVVV,"%s:%s(%d):Sending first response",__FILE__,__FUNCTION__,__LINE__);
	tx.frame_size = rx.frame_size;
	tx.n_frames = 0; // On first response, set number of frames to zero
	num_bytes_or_err = tx_frame(sockfd, (void *)&tx, sizeof(handshake));
	if (num_bytes_or_err <= 0) {
		log_message(RL_ERROR,"%s:%s(%d):Communication failure on completing handshake",__FILE__,__FUNCTION__,__LINE__);
		return num_bytes_or_err;
	}
	*nmem = rx.n_frames;
	*size = rx.frame_size;
	/* Receive data */
	if (rx.n_frames > 0) {
		log_message(RL_DEBUGVVV,"%s:%s(%d):Receiving data",__FILE__,__FUNCTION__,__LINE__);
		/* Create buffer to store expected data and set size parameters */
		*buf = malloc((*nmem) * (*size));
		for (frames_received=0; frames_received<*nmem; frames_received++) {
			num_bytes_or_err = rx_frame(sockfd, *buf + frames_received*(*size), *size);
			if (num_bytes_or_err <= 0) {
				log_message(RL_ERROR,"%s:%s(%d):Communication failure during data receiving",__FILE__,__FUNCTION__,__LINE__);
				return num_bytes_or_err;
			}
		}
	}
	/* Conclude communication by sending last response */
	log_message(RL_DEBUGVVV,"%s:%s(%d):Sending last response",__FILE__,__FUNCTION__,__LINE__);
	tx.frame_size = *size;
	tx.n_frames = *nmem; // On last response, set number of frames to all
	num_bytes_or_err = tx_frame(sockfd, (void *)&tx, sizeof(handshake));
	if (num_bytes_or_err <= 0) {
		log_message(RL_ERROR,"%s:%s(%d):Communication failure on sending last response",__FILE__,__FUNCTION__,__LINE__);
		return num_bytes_or_err;
	}
	log_message(RL_DEBUGVVV,"%s:%s(%d):Successful communication, received %lu frames",__FILE__,__FUNCTION__,__LINE__,*nmem);
	return 0;
}

int tx_frames(int sockfd, void *buf, ssize_t nmem, ssize_t size) {
	int num_bytes_or_err = 0;
	ssize_t frames_transmitted = 0;
	handshake tx = {
		.frame_size = size,
		.n_frames = nmem
	};
	handshake rx = {
		.frame_size = 0,
		.n_frames = nmem
	};
	/* Send handshake: frame size and number of frames */
	log_message(RL_DEBUGVVV,"%s:%s(%d):Initiating handshake",__FILE__,__FUNCTION__,__LINE__);
	num_bytes_or_err = tx_frame(sockfd,(void *)&tx, sizeof(handshake));
	if (num_bytes_or_err <= 0) {
		log_message(RL_ERROR,"%s:%s(%d):Communication failure on initiating handshake",__FILE__,__FUNCTION__,__LINE__);
		return num_bytes_or_err;
	}
	/* Wait for RX ready */
	log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting on first response",__FILE__,__FUNCTION__,__LINE__);
	num_bytes_or_err = rx_frame(sockfd,(void *)&rx, sizeof(handshake));
	if (num_bytes_or_err <= 0) {
		log_message(RL_ERROR,"%s:%s(%d):Communication failure waiting on first response",__FILE__,__FUNCTION__,__LINE__);
		return num_bytes_or_err;
	}
	/* Transmit data */
	if (rx.frame_size == tx.frame_size && rx.n_frames == 0) {
		log_message(RL_DEBUGVVV,"%s:%s(%d):Transmitting data",__FILE__,__FUNCTION__,__LINE__);
		for (frames_transmitted=0; frames_transmitted<nmem; frames_transmitted++) {
			num_bytes_or_err = tx_frame(sockfd,buf + frames_transmitted*size, size);
			if (num_bytes_or_err <= 0) {
				log_message(RL_ERROR,"%s:%s(%d):Communication failure during data transfer",__FILE__,__FUNCTION__,__LINE__);
				return num_bytes_or_err;
			}
		}
	} else {
		log_message(RL_WARNING,"%s:%s(%d):Invalid ACK received: frame_size = %u (expected %u), n_frames = %u (expected %u)",__FILE__,__FUNCTION__,__LINE__,rx.frame_size,size,rx.n_frames,0);
		return ERR_NET_INVALID_ACK;
	}
	/* Wait for RX ack all packets received */
	log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting on last response",__FILE__,__FUNCTION__,__LINE__);
	rx.frame_size = 0;
	rx.n_frames = 0;
	num_bytes_or_err = rx_frame(sockfd,(void *)&rx, sizeof(handshake));
	if (num_bytes_or_err <= 0) {
		log_message(RL_ERROR,"%s:%s(%d):Communication failure on last response",__FILE__,__FUNCTION__,__LINE__);
		return num_bytes_or_err;
	}
	if (rx.frame_size == tx.frame_size && rx.n_frames == nmem) {
		log_message(RL_DEBUGVVV,"%s:%s(%d):Successful communication, sent %lu frames",__FILE__,__FUNCTION__,__LINE__,nmem);
		return 0;
	} else {
		log_message(RL_WARNING,"%s:%s(%d):Invalid ACK received: frame_size = %u (expected %u), n_frames = %u (expected %u)",__FILE__,__FUNCTION__,__LINE__,rx.frame_size,size,rx.n_frames,nmem);
		return ERR_NET_INVALID_ACK;
	}
}
