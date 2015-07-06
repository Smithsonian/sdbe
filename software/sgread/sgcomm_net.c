#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <syslog.h>
#include <unistd.h>

#include <netinet/in.h>
#include <netdb.h> 

#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h> 

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
		//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Point of read",__FILE__,__FUNCTION__,__LINE__);
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
		} else {
			bytes_received += n;
			break;
		}
	} while (bytes_received < buflen);
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
