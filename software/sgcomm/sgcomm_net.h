#ifndef SGCOMM_NET_H
#define SGCOMM_NET_H

#include <netinet/in.h>

#define TIMEOUT_SEC 5
#define TIMEOUT_USEC 0
#define TIMEOUT_COUNT 3

typedef struct handshake_struct {
	ssize_t frame_size;
	ssize_t n_frames;
} handshake;

enum {
	ERR_NET_UNKNOWN_HOST = -2,
	ERR_NET_CANNOT_OPEN_SOCKET = -3,
	ERR_NET_CANNOT_CONNECT_SOCKET = -4,
	ERR_NET_CANNOT_BIND_SOCKET = -5,
	ERR_NET_CANNOT_LISTEN_ON_SOCKET = -6,
	ERR_NET_CANNOT_ACCEPT_CONNECTION = -7,
	ERR_NET_CANNOT_READ_SOCKET = -8,
	ERR_NET_TIMEOUT = -9,
	ERR_NET_CANNOT_WRITE_SOCKET = -10,
	ERR_NET_INVALID_ACK = -11
};

/* Initialize socket and connect to given host:port.
 * Arguments:
 *   const char *host -- Hostname or IP address
 *   uint16_t port -- Port number
 * Return:
 *   int -- File descriptor on success, or < 0 for error.
 * */
int make_socket_connect(const char *host, uint16_t port);

/* Initialize socket, bind to given host:port and listen for connection.
 * Arguments:
 *   const char *host -- Hostname or IP address
 *   uint16_t port -- Port number
 * Return:
 *   int -- File descriptor on success, or < 0 for error.
 * */
int make_socket_bind_listen(const char *host, uint16_t port);

/* Accept connection and return new socket.
 * Arguments:
 *   int sockfd -- File descriptor returned by make_socket_bind_listen()
 * Return:
 *   int -- File descriptor for connection on success, or < 0 for error.
 * */
int accept_connection(int sockfd);

/* Receive next frames transmission in auto-allocated buffer.
 * Arguments:
 *   sockfd -- Socket to receive data from
 *   void **buf -- Address of pointer to be used for referencing 
 *     allocated buffer
 *   ssize_t *nmem -- Pointer to memory for storing number of frames 
 *     received
 *   ssize_t * -- Pointer to memory for storing frame size
 * Return:
 *   int -- >= 0 and equal to the number of frames expected as indicated
 *     by sender, or < 0 for error (see enumerated error codes in 
 *     sgcomm_net.h).
 * Notes:
 *   This method expects one handshake structure which indicates the 
 *     size and number of frames to expect to initiate communication. A
 *     reply is sent in the form of another handshake structure which 
 *     acknowledges the frame size, but with the frame count set to 
 *     zero. At this point the method is ready and waits for data to be
 *     sent. After the total number of expected frames have been 
 *     received or if an error ocurred, communication is concluded by 
 *     sending another handshake structure with the correct frame size
 *     and the number of frames successfully received.
 *   A non-negative return value guarantees successful communication, 
 *     even though not ALL expected frames may have been received. It 
 *     is up to the calling method to check the number of received 
 *     frames agains the total number of expected frames, if necessary.
 *   The caller should free *buf when appropriate.
 * */
int rx_frames(int sockfd, void **buf, ssize_t *nmem, ssize_t *size);

/* Transmit a batch of frames.
 * Arguments:
 *   sockfd -- Socket to send data to
 *   void *buf -- Pointer to frame buffer in memory
 *   ssize_t nmem -- Number of frames to be sent
 *   ssize_t  -- Size of a frame in bytes
 * Return:
 *   int -- >= 0 and equal to the number of frames received at the other
 *     end, or < 0 for error (see enumerated error codes in 
 *     sgcomm_net.h).
 * Notes:
 *   This method initiates communication by first sending a handshake 
 *     structure that indicates the frame size and the number of frames
 *     to be transmitted. The receiver must reply with a similar 
 *     handshake structure which acknowledges the frame size, but with
 *     the frame counter zero. Once such a reply is received, data
 *     transmission begins. After all frames sent or error condition,
 *     a final handshake structure from the receiver is expected which
 *     again acknowledges the frame size and indicates the number of 
 *     frames successfully received.
 *   A non-negative return value guarantees successful communication, 
 *     even though not ALL frames may have been received at the other 
 *     end. It is up to the calling method to check the number of frames
 *     received agains the total number of frames, if necessary.
 * */
int tx_frames(int sockfd, void *buf, ssize_t nmem, ssize_t size);

#endif // SGCOMM_NET_H
