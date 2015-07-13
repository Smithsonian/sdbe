#ifndef SGCOMM_NET_H
#define SGCOMM_NET_H

#include <netinet/in.h>

#define TIMEOUT_SEC 1
#define TIMEOUT_USEC 0

typedef struct handshake_struct {
	uint32_t frame_size;
	uint32_t n_frames;
} handshake;

enum {
	ERR_NET_UNKNOWN_HOST = -2,
	ERR_NET_CANNOT_OPEN_SOCKET = -3,
	ERR_NET_CANNOT_CONNECT_SOCKET = -4,
	ERR_NET_CANNOT_BIND_SOCKET = -5,
	ERR_NET_CANNOT_LISTEN_ON_SOCKET = -6,
	ERR_NET_CANNOT_ACCEPT_CONNECTION = -7,
	ERR_NET_CANNOT_READ_SOCKET = -8,
	ERR_NET_READ_TIMEOUT = -9,
	ERR_NET_CANNOT_WRITE_SOCKET = -10,
	ERR_NET_INVALID_ACK = -11
};

int init_sockaddr(struct sockaddr_in *name,const char *hostname,uint16_t port);
int make_socket_connect(const char *host, uint16_t port);
int make_socket_bind_listen(const char *host, uint16_t port);
int accept_connection(int sockfd);
int rx_frame(int sockfd, void *buf, ssize_t buflen);
int tx_frame(int sockfd, void *buf, ssize_t buflen);
int rx_frames(int sockfd, void **buf, ssize_t *nmem, ssize_t *size);
int tx_frames(int sockfd, void *buf, ssize_t nmem, ssize_t size);

#endif // SGCOMM_NET_H
