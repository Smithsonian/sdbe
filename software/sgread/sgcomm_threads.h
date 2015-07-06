#ifndef SGCOMM_THREADS_H
#define SGCOMM_THREADS_H

#include "sgcomm_report.h"

#define WAIT_PERIOD_US (10000 + rand()%1000)

/* States that define thread behaviour. CTRL_STATE_STOP and higher are
 * conditions that may cause the thread to ignore any control state
 * settings; proceed with caution. */
typedef enum {
	CS_INIT, // keep INIT lowest value
	CS_START,
	CS_RUN,
	CS_WAIT,
	CS_STOP,
	CS_DONE,
	CS_ERROR // keep ERROR highest value
} ctrl_state;

/* Enumeration of pthread types for easier control */
typedef enum {
	TT_MAIN,
	TT_READER,
	TT_TRANSMITTER,
	TT_RECEIVER,
	TT_WRITER,
	TT_CONTROLLER,
	TT_NUM_TYPES // always last, counter for number of thread types
} thread_type;

typedef struct sgcomm_thread_struct {
	thread_type type;
	ctrl_state state;
	pthread_mutex_t mtx;
	pthread_t thread;
	void *type_msg;
	int (*destroy_message)(void *type_msg);
	int (*get_status)(status_update *su);
	void *(*run_method)(void *arg);
} sgcomm_thread;

/* Data buffer shared between reader / transmitter threads. */
typedef struct shared_buffer_struct {
	uint32_t *buf; // Where data (VDIF frames) are stored
	size_t buf_size; // Size of buffer in sizeof(uint32_t)
	uint32_t frame_size; // Size of a single data unit (VDIF frame) in sizeof(uint32_t)
	uint32_t n_frames; // Number of data units (VDIF frames) available in buffer
	pthread_mutex_t mtx; // Mutex to handle multi-threaded access
} shared_buffer;

/* Message structure to initialize a reader thread */
typedef struct reader_msg_struct {
	/* Arguments needed to initialize read-mode SGPlan */
	char *pattern;
	char *fmtstr;
	int *mod_list;
	int n_mod;
	int *disk_list;
	int n_disk;
	/* Other parameters needed? */
	shared_buffer *dest;
} reader_msg;

/* Message structure to initialize a transmitter thread */
typedef struct transmitter_msg_struct {
	/* Other parameters needed? */
	shared_buffer *src;
	// connection information?
	char *host;
	uint16_t port;
} transmitter_msg;

/* Message structure to initialize a receiver thread */
typedef struct receiver_msg_struct {
	/* Other parameters needed? */
	shared_buffer *dest;
	// connection information?
	char *host;
	uint16_t port;
} receiver_msg;

/* Message structure to initialize a writer thread */
typedef struct writer_msg_struct {
	/* Arguments needed to initialize read-mode SGPlan */
	char *pattern;
	char *fmtstr;
	int *mod_list;
	int n_mod;
	int *disk_list;
	int n_disk;
	/* Other parameters needed? */
	shared_buffer *src;
} writer_msg;

/* Thread general */
sgcomm_thread * create_thread(thread_type tt);
int start_thread(sgcomm_thread *st);
int stop_thread(sgcomm_thread *st);
char *get_thread_type_str(thread_type tt);
int set_thread_state(sgcomm_thread *st, ctrl_state state, 
						const char *msg, ... );
int get_thread_state(sgcomm_thread *st, ctrl_state *state);
int destroy_thread(sgcomm_thread **st);

/* Thread type specific */
int init_reader_msg(reader_msg *msg, shared_buffer *dest,
					const char *pattern, const char *fmtstr, 
					const int *mod_list, int n_mod, 
					const int *disk_list, int n_disk);
int init_transmitter_msg(transmitter_msg *msg, shared_buffer *src, 
							const char *host, uint16_t port);
int init_receiver_msg(receiver_msg *msg, shared_buffer *dest, 
							const char *host, uint16_t port);
int init_writer_msg(writer_msg *msg, shared_buffer *src,
					const char *pattern, const char *fmtstr, 
					const int *mod_list, int n_mod, 
					const int *disk_list, int n_disk);

/* Shared buffer */
shared_buffer * create_shared_buffer(size_t buffer_size);
int destroy_shared_buffer(shared_buffer **sb);
int obtain_data_lock(shared_buffer *sb);
int release_data_lock(shared_buffer *sb);

#endif // SGCOMM_THREADS_H
