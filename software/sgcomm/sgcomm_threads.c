#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "sgcomm_net.h"
#include "sgcomm_report.h"
#include "sgcomm_threads.h"

#include "scatgat.h"

/* Data reading from scatter-gather files */
static int _get_status_reader(status_update *su);
static int _destroy_reader_msg(void *type_msg);
static void * _threaded_reader(void *arg);
/* Data transmitting over 10GbE */
static int _get_status_transmitter(status_update *su);
static int _destroy_transmitter_msg(void *type_msg);
static void * _threaded_transmitter(void *arg);
/* Data receiving over 10GbE */
static int _get_status_receiver(status_update *su);
static int _destroy_receiver_msg(void *type_msg);
static void * _threaded_receiver(void *arg);
/* Data writing to scatter-gather files */
static int _get_status_writer(status_update *su);
static int _destroy_writer_msg(void *type_msg);
static void * _threaded_writer(void *arg);
/* Control flow */
static int _get_status_controller(status_update *su);
static void * _threaded_controller(void *arg);

sgcomm_thread * create_thread(thread_type tt) {
	sgcomm_thread *st = malloc(sizeof(sgcomm_thread));
	st->type = tt;
	switch(tt) {
	case TT_READER:
		st->type_msg = (reader_msg *)malloc(sizeof(reader_msg));
		st->destroy_message = &_destroy_reader_msg;
		st->get_status = &_get_status_reader;
		st->run_method = &_threaded_reader;
		break;
	case TT_TRANSMITTER:
		st->type_msg = (transmitter_msg *)malloc(sizeof(transmitter_msg));
		st->destroy_message = &_destroy_transmitter_msg;
		st->get_status = &_get_status_transmitter;
		st->run_method = &_threaded_transmitter;
		break;
	case TT_RECEIVER:
		st->type_msg = (receiver_msg *)malloc(sizeof(receiver_msg));
		st->destroy_message = &_destroy_receiver_msg;
		st->get_status = &_get_status_receiver;
		st->run_method = &_threaded_receiver;
		break;
	case TT_WRITER:
		st->type_msg = (writer_msg *)malloc(sizeof(writer_msg));
		st->destroy_message = &_destroy_writer_msg;
		st->get_status = &_get_status_writer;
		st->run_method = &_threaded_writer;
		break;
	default:
		if (st != NULL) {
			free(st);
			st = NULL;
		}
		return st;
	}
	/* Check if message space allocated */
	if (st != NULL) {
		if (st->type_msg == NULL) {
			free(st);
			st = NULL;
		}
	}
	/* Initialize the mutex */
	pthread_mutex_init(&(st->mtx),NULL);
	set_thread_state(st, CS_INIT, "%s:%s(%d):Initialized %s thread",__FILE__,__FUNCTION__,__LINE__,get_thread_type_str(st->type));
	return st;
}

int destroy_thread(sgcomm_thread **st) {
	if ((*st)->destroy_message((*st)->type_msg) != 0)
		return -1;
	pthread_mutex_destroy(&((*st)->mtx));
	free(*st);
	st = NULL;
	return 0;
}

int start_thread(sgcomm_thread *st) {
	log_message(RL_DEBUGVVV,"%s:%s(%d):Starting thread",__FILE__,__FUNCTION__,__LINE__);
	if (st->run_method == NULL || st == NULL)
		return -1;
	return pthread_create(&(st->thread),NULL,st->run_method,st);
}

int stop_thread(sgcomm_thread *st) {
	set_thread_state(st, CS_STOP, NULL);
	return pthread_join(st->thread,NULL);
}

char *get_thread_type_str(thread_type tt) {
	switch(tt) {
		case TT_MAIN:
			return "TT_MAIN";
		case TT_READER:
			return "TT_READER";
		case TT_TRANSMITTER:
			return "TT_TRANSMITTER";
		case TT_RECEIVER:
			return "TT_RECEIVER";
		case TT_WRITER:
			return "TT_WRITER";
		case TT_CONTROLLER:
			return "TT_CONTROLLER";
		case TT_NUM_TYPES:
			return "TT_NUM_TYPES";
		default:
			return "TT_INVALID";
	}
}

shared_buffer * create_shared_buffer(size_t buffer_size) {
	shared_buffer *sb = malloc(sizeof(shared_buffer));
	sb->buf = (uint32_t *)malloc(buffer_size*sizeof(uint32_t));
	if (sb->buf == NULL) {
		free(sb);
		sb = NULL;
	}
	sb->buf_size = buffer_size;
	sb->n_frames = 0;
	sb->frame_size = 0;
	pthread_mutex_init(&(sb->mtx),NULL);
	return sb;
}

int destroy_shared_buffer(shared_buffer **sb) {
	shared_buffer *sb_cpy = *sb;
	if (obtain_data_lock(sb_cpy) == 0) {
		/* Set external references to shared buffer NULL and release
		 * mutex */
		*sb = NULL;
		release_data_lock(sb_cpy);
		/* Free resources using copy reference */
		if (sb_cpy->buf != NULL)
			free(sb_cpy->buf);
		pthread_mutex_destroy(&(sb_cpy->mtx));
		free(sb_cpy);
		return 0;
	}
	return -1;
}

int obtain_data_lock(shared_buffer *sb) {
	//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting on data",__FILE__,__FUNCTION__,__LINE__);
	if (pthread_mutex_lock(&(sb->mtx)) == 0)
		return 0;
	log_message(RL_ERROR,"%s:%s(%d):Could not lock mutex for shared_buffer %lu",__FILE__,__FUNCTION__,__LINE__,sb);
	return -1;
}

int release_data_lock(shared_buffer *sb) {
	if (pthread_mutex_unlock(&(sb->mtx)) == 0)
		return 0;
	log_message(RL_ERROR,"%s:%s(%d):Could not unlock mutex for shared_buffer %lu",__FILE__,__FUNCTION__,__LINE__,sb);
	return -1;
}

int set_thread_state(sgcomm_thread *st, ctrl_state state, const char *fmt, ...) {
	va_list ap;
	int result;
	report_level rl;
	
	//~ log_message(RL_DEBUGVVV,"%s:Enter",__FUNCTION__);
	
	va_start(ap, fmt);
	if (state > CS_ERROR || state < CS_INIT) {
		log_message(RL_ERROR,"%s:%s(%d):Invalid ctrl_state %d",__FILE__,__FUNCTION__,__LINE__,(int)state);
		result = -1;
	}
	
	//~ log_message(RL_DEBUGVVV,"%s:Given state okay",__FUNCTION__);
	
	if (pthread_mutex_lock(&(st->mtx)) == 0) {
		st->state = state;
		if (pthread_mutex_unlock(&(st->mtx)) != 0) {
			log_message(RL_ERROR,"%s:s(%d):Cannot unlock mutex for thread %s (%lu)",__FILE__,__FUNCTION__,__LINE__,get_thread_type_str(st->type),st->thread);
			result = -1;
		}
	} else {
		log_message(RL_ERROR,"%s:%s(%d):Cannot lock mutex for thread %s (%lu)",__FILE__,__FUNCTION__,__LINE__,get_thread_type_str(st->type),st->thread);
		result = -1;
	}
	
	//~ log_message(RL_DEBUGVVV,"%s:State is set",__FUNCTION__);
	
	switch (state) {
	case CS_ERROR:
		rl = RL_ERROR;
		break;
	case CS_RUN:
	case CS_WAIT:
		rl = RL_INFO;
	default:
		rl = RL_NOTICE;
	}
	
	//~ log_message(RL_DEBUGVVV,"%s:Log level is set",__FUNCTION__);
	
	vlog_message(rl,fmt,ap);
	result = 0;
	va_end(ap);
	
	//~ log_message(RL_DEBUGVVV,"%s:Leave",__FUNCTION__);
	
	return result;
}

int get_thread_state(sgcomm_thread *st, ctrl_state *state) {
	int result;
	
	if (pthread_mutex_lock(&(st->mtx)) == 0) {
		*state = st->state;
		if (pthread_mutex_unlock(&(st->mtx)) != 0) {
			log_message(RL_ERROR,"%s:%s(%d):Unable to unlock mutex for thread %s (%lu)",__FILE__,__FUNCTION__,__LINE__,get_thread_type_str(st->type),st->thread);
			result = -1;
		}
	} else {
		log_message(RL_ERROR,"%s:%s(%d):Unable to lock mutex for thread %s (%lu)",__FILE__,__FUNCTION__,__LINE__,get_thread_type_str(st->type),st->thread);
		result = -1;
	}
	result = 0;
	
	return result;
}

void _print_reader_msg(reader_msg *msg) {
	char str_disk_list[0x100], str_mod_list[0x100];
	int ii;
	for (ii=0; ii<msg->n_disk; ii++)
		snprintf(str_disk_list+3*ii,0x100-3*ii-1,"%3d",msg->disk_list[ii]);
	for (ii=0; ii<msg->n_mod; ii++)
		snprintf(str_mod_list+3*ii,0x100-3*ii-1,"%3d",msg->mod_list[ii]);
	fprintf(stdout,	"reader_msg:\n"
					"  matches %s x ([%s],[%s],%s)\n",
					msg->fmtstr,str_mod_list,str_disk_list,msg->pattern);
}

int init_reader_msg(reader_msg *msg, shared_buffer *dest,
					const char *pattern, const char *fmtstr, 
					const int *mod_list, int n_mod, 
					const int *disk_list, int n_disk) {
	if (msg == NULL || dest == NULL ||
		pattern == NULL || fmtstr == NULL || 
		mod_list == NULL || disk_list == NULL)
		return -1;
	if (n_mod == 0 || n_disk == 0)
		return -1;
	//~ fprintf(stdout,"pattern: '%s'\n",pattern);
	//~ fprintf(stdout,"fmtstr: '%s'\n",fmtstr);
	msg->dest = dest;
	msg->pattern = (char *)malloc(strlen(pattern)+1);
	strcpy(msg->pattern,pattern);
	msg->fmtstr = (char *)malloc(strlen(fmtstr)+1);
	strcpy(msg->fmtstr,fmtstr);
	msg->mod_list = (int *)malloc(n_mod*sizeof(int));
	memcpy(msg->mod_list, mod_list, n_mod*sizeof(int));
	msg->n_mod = n_mod;
	msg->disk_list = (int *)malloc(n_disk*sizeof(int));
	memcpy(msg->disk_list, disk_list, n_disk*sizeof(int));
	msg->n_disk = n_disk;
	//~ _print_reader_msg(msg);
	return 0;
}


int _destroy_reader_msg(void *type_msg) {
	reader_msg *msg = (reader_msg *)type_msg;
	if (msg->pattern != NULL)
		free(msg->pattern);
	if (msg->fmtstr != NULL)
		free(msg->fmtstr);
	if (msg->mod_list != NULL)
		free(msg->mod_list);
	if (msg->disk_list != NULL)
		free(msg->disk_list);
	return 0;
}

int init_transmitter_msg(transmitter_msg *msg, shared_buffer *src, 
							const char *host, uint16_t port) {
	if (msg == NULL || src == NULL)
		return -1;
	msg->src = src;
	msg->host = (char *)malloc(strlen(host)+1);
	strcpy(msg->host,host);
	msg->port = port;
	return 0;
}

int _destroy_transmitter_msg(void *type_msg) {
	transmitter_msg *msg = (transmitter_msg *)type_msg;
	if (msg->host != NULL)
		free(msg->host);
	return 0;
}

int init_receiver_msg(receiver_msg *msg, shared_buffer *dest, 
							const char *host, uint16_t port) {
	if (msg == NULL || dest == NULL)
		return -1;
	msg->dest = dest;
	msg->host = (char *)malloc(strlen(host)+1);
	strcpy(msg->host,host);
	msg->port = port;
	return 0;
}

int _destroy_receiver_msg(void *type_msg) {
	receiver_msg *msg = (receiver_msg *)type_msg;
	if (msg->host != NULL)
		free(msg->host);
	return 0;
}

int init_writer_msg(writer_msg *msg, shared_buffer *src,
					const char *pattern, const char *fmtstr, 
					const int *mod_list, int n_mod, 
					const int *disk_list, int n_disk) {
	if (msg == NULL || src == NULL ||
		pattern == NULL || fmtstr == NULL || 
		mod_list == NULL || disk_list == NULL)
		return -1;
	if (n_mod == 0 || n_disk == 0)
		return -1;
	msg->src = src;
	msg->pattern = (char *)malloc(strlen(pattern)+1);
	strcpy(msg->pattern,pattern);
	msg->fmtstr = (char *)malloc(strlen(fmtstr)+1);
	strcpy(msg->fmtstr,fmtstr);
	msg->mod_list = (int *)malloc(n_mod*sizeof(int));
	memcpy(msg->mod_list, mod_list, n_mod*sizeof(int));
	msg->n_mod = n_mod;
	msg->disk_list = (int *)malloc(n_disk*sizeof(int));
	memcpy(msg->disk_list, disk_list, n_disk*sizeof(int));
	msg->n_disk = n_disk;
	return 0;
}

int _destroy_writer_msg(void *type_msg) {
	writer_msg *msg = (writer_msg *)type_msg;
	if (msg->pattern != NULL)
		free(msg->pattern);
	if (msg->fmtstr != NULL)
		free(msg->fmtstr);
	if (msg->mod_list != NULL)
		free(msg->mod_list);
	if (msg->disk_list != NULL)
		free(msg->disk_list);
	return 0;
}

static void * _threaded_reader(void *arg) {
	int n_sg; // number of scatter gather files
	SGPlan *sgpln; // SGPlan for reading data
	uint32_t *local_buf = NULL; // local reader buffer
	int n_frames = 0; // number of frames available in buffer
	int n_frames_copied = 0; // number of frames copied from local to shared buffer
	int n_frames_this_copy = 0; // number for frames copied per iteration
	int wait_after_data = 0;
	
	sgcomm_thread *st = (sgcomm_thread *)arg; // general thread message
	reader_msg *msg = (reader_msg *)(st->type_msg); // type specific message
	shared_buffer *dest = msg->dest;
	ctrl_state ctrl;
	
	/* Set this thread just started (i.e. not yet in loop) */
	set_thread_state(st, CS_START, "%s:%s(%d):Thread started",__FILE__,__FUNCTION__,__LINE__);
	
	/* Make scatter-gather plan */
	n_sg = make_sg_read_plan(&sgpln, msg->pattern, msg->fmtstr, msg->mod_list, msg->n_mod, msg->disk_list, msg->n_disk); // REDO: n_sg = 1;
	if (n_sg <= 0)
		set_thread_state(st, CS_ERROR, "%s:%s(%d):Read-mode SGPlan failed, returned %d",__FILE__,__FUNCTION__,__LINE__,n_sg);
	else {
		log_message(RL_INFO,"%s:%s(%d):Read-mode SGPlan created from %d files",__FILE__,__FUNCTION__,__LINE__,n_sg);
		
		/* From scatter-gather plan we can set frame size */
		if (obtain_data_lock(dest) == 0) {
			dest->frame_size = sgpln->sgprt[0].sgi->pkt_size/sizeof(uint32_t); // REDO: dest->frame_size = 1;
			if (release_data_lock(dest) != 0)
				set_thread_state(st, CS_ERROR, "%s:%s(%d):Cannot release shared buffer",__FILE__,__FUNCTION__,__LINE__);
		} else
			set_thread_state(st, CS_ERROR, "%s:%s(%d):Cannot access shared buffer",__FILE__,__FUNCTION__,__LINE__);
		
		log_message(RL_INFO,"%s:%s(%d):Frame size set to %u x sizeof(uint32_t)",__FILE__,__FUNCTION__,__LINE__,dest->frame_size);
	}
	
	/* Set this thread entering infinite loop section */
	if (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP)) {
		set_thread_state(st, CS_RUN, NULL);
		
		log_message(RL_DEBUG,"%s:%s(%d):Thread set to enter loop",__FILE__,__FUNCTION__,__LINE__);
	}
	
	/* Continue working until stop condition */
	while (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP)) {
		/* If this thread is in a wait state, sleep and repeat check */
		if (ctrl == CS_WAIT) {
			log_message(RL_DEBUGVVV,"%s:%s(%d):Waiting due to CS_WAIT",__FILE__,__FUNCTION__,__LINE__);
			usleep(WAIT_PERIOD_US);
			continue;
		}
		
		/* Read data into local buffer, only if n_frames == 0 which 
		 * means that data from a previous read was successfully passed
		 * to an empty shared buffer. In case n_frames != 0, then add a
		 * once-off sleep for one wait period before attempting to pass
		 * data again to shared buffer. */
		if (n_frames == 0) {
			n_frames = read_next_block_vdif_frames(sgpln, &local_buf); // REDO: n_frames = dest->buf_size/dest->frame_size; n_frames_copied = 0; local_buf = (uint32_t *)malloc(n_frames*dest->frame_size*sizeof(uint32_t)); for (int ii=0; ii<n_frames; ii++) local_buf[ii] = (uint32_t)ii;
			/* If number of frames read is non-positive, cannot continue
			 * running */
			if (n_frames < 0) {
				set_thread_state(st,CS_ERROR,"%s:%s(%d):Reading from SGPlan failed, returned %d",__FILE__,__FUNCTION__,__LINE__,n_frames);
				break;
			} else if (n_frames == 0) {
				set_thread_state(st,CS_STOP,"%s:%s(%d):End of scatter-gather reached, returned %d",__FILE__,__FUNCTION__,__LINE__,n_frames);
				break;
			}
			
			log_message(RL_DEBUGVVV,"%s:%s(%d):Read %d frames",__FILE__,__FUNCTION__,__LINE__,n_frames);
		}
		
		/* If there are frames to process, insert into shared buffer */
		if (obtain_data_lock(dest) == 0) {
			/* Only add frames if shared buffer is empty */
			if (dest->n_frames == 0) {
				/* Calculate copy size, all frames if possible,
				 * otherwise as many as the shared data buffer can keep */
				n_frames_this_copy = (n_frames-n_frames_copied)*dest->frame_size < dest->buf_size ? 
					(n_frames-n_frames_copied) : 
					(int)(dest->buf_size/dest->frame_size);
				
				log_message(RL_DEBUGVVV,"%s:%s(%d):Copy %d/%d frames to shared buffer (buffer allows %d)",__FILE__,__FUNCTION__,__LINE__,n_frames_this_copy,n_frames,(int)(dest->buf_size/dest->frame_size));
				
				/* Copy data to shared buffer and set frame count.
				 * Source location is offset by the number of frames 
				 * that has been copied so far */
				memcpy(dest->buf, &(local_buf[n_frames_copied*dest->frame_size]),
						n_frames_this_copy*dest->frame_size*sizeof(uint32_t));
				/* Update the frame counter in the shared buffer and 
				 * increment the total number of frames copied */
				dest->n_frames = n_frames_this_copy;
				n_frames_copied += n_frames_this_copy;
				
				log_message(RL_DEBUGVVV,"%s:%s(%d):Copied %u (%u/%u) frames of data [%u .. %u] into shared buffer",__FILE__,__FUNCTION__,__LINE__,n_frames_this_copy,n_frames_copied,n_frames,dest->buf[0],dest->buf[dest->n_frames*dest->frame_size-1]);
				
				/* If all the frames have been copied, reset frames 
				 * available and free the local buffer */
				if (n_frames_copied == n_frames) {
					free(local_buf);
					local_buf = NULL;
					n_frames = 0;
					n_frames_copied = 0;
				}
				/* If shared buffer was empty, optimistic that we don't 
				 * have to wait on next iteration */
				wait_after_data = 0;
			} else {
				wait_after_data = 1;
				//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Shared buffer not empty, setting wait state",__FILE__,__FUNCTION__,__LINE__);
			}
			
			/* Finally, release lock on shared data */
			if (release_data_lock(dest) != 0) {
				set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not release shared buffer",__FILE__,__FUNCTION__,__LINE__);
				break;
			}
			if (wait_after_data)
				usleep(WAIT_PERIOD_US);
		} else
			set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not access shared buffer",__FILE__,__FUNCTION__,__LINE__);
		
		// TODO: Add other reader main loop tasks here, outside the
		// data lock, e.g. update status, etc?
		
		//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Repeat loop",__FILE__,__FUNCTION__,__LINE__);
	}
	
	log_message(RL_DEBUG,"%s:%s(%d):Thread exited main loop",__FILE__,__FUNCTION__,__LINE__);
	
	/* In case we exited without properly handling this memory, as is 
	 * the case when frame request returned non-positive number. */
	if (local_buf != NULL) {
		free(local_buf);
		local_buf = NULL;
	}
	
	close_sg_read_plan(sgpln);
	
	set_thread_state(st, CS_DONE,"%s:%s(%d):Thread is done",__FILE__,__FUNCTION__,__LINE__);
	return NULL;
}

static void * _threaded_transmitter(void *arg) {
	int frames_sent = 0;
	int sockfd;
	int wait_after_data = 0;
	
	/* For new communication protocol */
	#define MAX_TX_TRIES 3
	int tx_tries;
	
	sgcomm_thread *st = (sgcomm_thread *)arg; // general thread message
	transmitter_msg *msg = (transmitter_msg *)(st->type_msg); // type specific message
	shared_buffer *src = msg->src;
	ctrl_state ctrl;
	
	/* Set this thread just started, i.e. not yet in main loop */
	set_thread_state(st, CS_START,"%s:%s(%d):Thread started",__FILE__,__FUNCTION__,__LINE__);
	
	/* Create socket and connect */
	sockfd = make_socket_connect(msg->host,msg->port);
	if (sockfd < 0)
		set_thread_state(st, CS_ERROR,"%s:%s(%d):Cannot connect socket",__FILE__,__FUNCTION__,__LINE__);
	
	/* Set this thread entering infinite loop section */
	if (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP)) {
		set_thread_state(st, CS_RUN,"%s:%s(%d):Connected",__FILE__,__FUNCTION__,__LINE__);
		
		log_message(RL_DEBUG,"%s:%s(%d):Thread set to enter loop",__FILE__,__FUNCTION__,__LINE__);
	}
	
	while (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP)) {
		/* If this thread is in a wait state, sleep and repeat check */
		if (ctrl == CS_WAIT) {
			usleep(WAIT_PERIOD_US);
			continue;
		}
		
		//~ usleep(500000);
		
		/* Check if data available */
		if (obtain_data_lock(src) == 0) {
			if (src->n_frames > 0) {
				
				log_message(RL_DEBUGVVV,"%s:%s(%d):Send %u frames of data [%u .. %u] from shared buffer",__FILE__,__FUNCTION__,__LINE__,src->n_frames,src->buf[0],src->buf[src->n_frames*src->frame_size-1]);
				
				// TODO: Process data, send, etc
				/*******************************************************
				 * This section is being replaced to accommodate a new
				 * communication protocol
				 * 
				frames_sent = 0;
				do {
					if (tx_frame(sockfd,(void *)(src->buf+frames_sent*src->frame_size),(src->frame_size)*sizeof(uint32_t)) < 0) {
						set_thread_state(st, CS_ERROR,"%s:%s(%d):Cannot transmit frame",__FILE__,__FUNCTION__,__LINE__);
						break;
					}
				} while (++frames_sent < src->n_frames);
				/* 
				* *****************************************************
				* */
				frames_sent = 0;
				tx_tries = MAX_TX_TRIES;
				do {
					if (tx_frames(sockfd, (void *)(src->buf), src->n_frames, src->frame_size*sizeof(uint32_t)) == 0)
						break;
					log_message(RL_WARNING,"%s:%s(%d):Sending frames failed, %d tries left",__FILE__,__FUNCTION__,__LINE__,tx_tries);
				} while (tx_tries-- > 0);
				if (tx_tries > 0) {
					frames_sent = src->n_frames;
				} else {
					set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not send frames, stopping",__FILE__,__FUNCTION__,__LINE__);
					frames_sent = 0;
				}
				/*
				 * ****************************************************/
				log_message(RL_DEBUGVVV,"%s:%s(%d):Sent %u frames",__FILE__,__FUNCTION__,__LINE__,frames_sent);
				
				/* Set frame count to zero */
				src->n_frames = 0;
				/* If there was data, optimistic that we don't have to 
				 * wait on next iteration */
				wait_after_data = 0;
			} else {
				wait_after_data = 1;
				//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Shared buffer empty, set wait state",__FILE__,__FUNCTION__,__LINE__);
			}
			
			/* Finally, release lock on shared data */
			if (release_data_lock(src) != 0) {
				set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not release shared buffer",__FILE__,__FUNCTION__,__LINE__);
				break;
			}
			
			if (wait_after_data)
				usleep(WAIT_PERIOD_US);
		} else
			set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not access shared buffer",__FILE__,__FUNCTION__,__LINE__);
		
		//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Repeat loop",__FILE__,__FUNCTION__,__LINE__);
	}
	
	log_message(RL_DEBUG,"%s:%s(%d):Thread exited main loop",__FILE__,__FUNCTION__,__LINE__);
	
	/* Part of new communication protocol, send zero frames to end */
	if (tx_frames(sockfd, (void *)(src->buf), src->n_frames, src->frame_size*sizeof(uint32_t)) != 0)
		log_message(RL_WARNING,"%s:%s(%d):Could not send end-of-transmission",__FILE__,__FUNCTION__,__LINE__);
	
	// TODO: Close socket, etc
	if (sockfd >= 0)
		close(sockfd);
	
	set_thread_state(st, CS_DONE,"%s:%s(%d):Thread is done",__FILE__,__FUNCTION__,__LINE__,get_thread_type_str(st->type));
	return NULL;
}

static void * _threaded_receiver(void *arg) {
	int frames_received = 0;
	int frame_size = 0;
	int num_bytes_or_err = 0;
	int max_frames_in_buffer = 0;
	uint32_t *local_buf = NULL;
	int sockfd_listen;
	int sockfd_receive;
	int wait_after_data = 0;
	
	/* Required for new communication protocol */
	ssize_t nmem, size, frames_to_copy, tmp_buf_offset = 0;
	void *tmp_buf = NULL;
	
	sgcomm_thread *st = (sgcomm_thread *)arg; // general thread message
	receiver_msg *msg = (receiver_msg *)(st->type_msg); // type specific message
	shared_buffer *dest = msg->dest;
	ctrl_state ctrl;
	
	/* Set this thread just started, i.e. not yet in main loop */
	set_thread_state(st, CS_START,"%s:%s(%d):Thread started",__FILE__,__FUNCTION__,__LINE__);
	
	/* Create socket and listen */
	sockfd_listen = make_socket_bind_listen(msg->host,msg->port);
	if (sockfd_listen < 0) 
		set_thread_state(st, CS_ERROR,"%s:%s(%d):Cannot connect socket",__FILE__,__FUNCTION__,__LINE__);
	
	if (obtain_data_lock(dest) == 0) {
		dest->frame_size = 264; // REDO: We need some other way to set the frame size
		frame_size = dest->frame_size;
		max_frames_in_buffer = (uint32_t)dest->buf_size/frame_size;
		local_buf = (uint32_t *)malloc(max_frames_in_buffer*frame_size*sizeof(uint32_t));
		if (release_data_lock(dest) != 0)
			set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not release shared buffer",__FILE__,__FUNCTION__,__LINE__);
	} else
		set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not access shared buffer",__FILE__,__FUNCTION__,__LINE__);
	
	/* Set this thread entering infinite loop section */
	if (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP)) {
		/* Block until connection */
		sockfd_receive = accept_connection(sockfd_listen);
		if (sockfd_receive < 0)
			set_thread_state(st, CS_ERROR,"%s:%s(%d):Cannot accept connection",__FILE__,__FUNCTION__,__LINE__);
		/* And now ready to run */
		set_thread_state(st, CS_RUN,"%s:%s(%d):Ready to receive data",__FILE__,__FUNCTION__,__LINE__);
		log_message(RL_DEBUG,"%s:%s(%d):Thread set to enter loop",__FILE__,__FUNCTION__,__LINE__);
	}
	
	while (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP)) {
		/* If this thread is in a wait state, sleep and repeat check */
		if (ctrl == CS_WAIT) {
			usleep(WAIT_PERIOD_US);
			continue;
		}
		
		/* Receive data into local buffer */
		// TODO: Need some kind of timeout condition here
		/***************************************************************
		 * This section is being replaced to accommodate a more 
		 * sophisticated communication protocol.
		 * *
		if (frames_received == 0) {
			//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Entering rx_frame loop",__FILE__,__FUNCTION__,__LINE__);
			do {
				num_bytes_or_err = rx_frame(sockfd_receive,(void *)(local_buf + frames_received*frame_size), frame_size*sizeof(uint32_t));
				if (num_bytes_or_err <= 0) {
					if (num_bytes_or_err == ERR_NET_READ_TIMEOUT || num_bytes_or_err == 0) {
						set_thread_state(st, CS_STOP,"%s:%s(%d):Apparently reached end-of-stream",__FILE__,__FUNCTION__,__LINE__);
					} else {
						set_thread_state(st, CS_ERROR,"%s:%s(%d):Cannot receive frame",__FILE__,__FUNCTION__,__LINE__);
					}
					break;
				}
				
				//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Received %u frames",__FILE__,__FUNCTION__,__LINE__,frames_received+1);
			} while (++frames_received < max_frames_in_buffer);
		/* 
		* *************************************************************
		* */
		if (frames_received < max_frames_in_buffer) {
			do {
				if (tmp_buf == NULL) {
					if (rx_frames(sockfd_receive,&tmp_buf,&nmem,&size) != 0)
						set_thread_state(st, CS_RUN,"%s:%s(%d):Error on receiving data",__FILE__,__FUNCTION__,__LINE__);
				}
				/* If zero frames received, that's end-of-transmission */
				if (nmem == 0) {
					set_thread_state(st, CS_STOP,"%s:%s(%d):End-of-transmission received",__FILE__,__FUNCTION__,__LINE__);
					break;
				}
				frames_to_copy = frames_received + nmem < max_frames_in_buffer ? nmem : max_frames_in_buffer-frames_received;
				memcpy((void *)(local_buf+frames_received*frame_size),tmp_buf+tmp_buf_offset,frames_to_copy*size);
				frames_received += frames_to_copy;
				if (frames_to_copy == nmem) {
					free(tmp_buf);
					tmp_buf = NULL;
					tmp_buf_offset = 0;
				} else {
					tmp_buf_offset += frames_to_copy;
					nmem -= frames_to_copy;
				}
			} while(frames_received < max_frames_in_buffer);
			
		/* 
		 * End of replacement to accommodate new communication protocol.
		 * ************************************************************/
			
			log_message(RL_DEBUGVVV,"%s:%s(%d):Received %u frames",__FILE__,__FUNCTION__,__LINE__,frames_received);
			
		} else
			usleep(WAIT_PERIOD_US);
		
		/***************************************************************
		 * Remainder of the main loop was within the 
		 *   if (frames_received == 0) {
		 * case (before the new communication protocol change), which 
		 * may have cause a deadlock condition if the receiver thread
		 * had to wait for the writer thread.
		 * ************************************************************/
		 
		/* Store data in shared buffer. Need to check if there is 
		 * an error condition, as would be the case if something 
		 * went wrong with receiving. Note that CS_STOP is allowed 
		 * to continue since there may be some frames left to write 
		 * in that case */
		//~ log_message(RL_DEBUGVVV,"%s:%s(%d):                                             Try to get data lock...",__FILE__,__FUNCTION__,__LINE__);
		if (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_ERROR) && obtain_data_lock(dest) == 0) {
			if (frames_received > 0 && dest->n_frames == 0) {
				memcpy(dest->buf, local_buf, frames_received*frame_size*sizeof(uint32_t));
				dest->n_frames = frames_received;
				
				log_message(RL_DEBUGVVV,"%s:%s(%d):Copied %u frames of data [%u .. %u] into shared buffer",__FILE__,__FUNCTION__,__LINE__,frames_received,dest->buf[0],dest->buf[dest->n_frames*dest->frame_size-1]);
				/* If there was data, optimistic that we don't have to 
				* wait on next iteration */
				wait_after_data = 0;
				frames_received = 0;
			} else {
				wait_after_data = 1;
				//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Shared buffer not empty, set wait state",__FILE__,__FUNCTION__,__LINE__);
			}
			/* Finally, release lock on shared data */
			if (release_data_lock(dest) != 0) {
				set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not release shared buffer",__FILE__,__FUNCTION__,__LINE__);
				break;
			}
			//~ log_message(RL_DEBUGVVV,"%s:%s(%d):                                         ...released data lock",__FILE__,__FUNCTION__,__LINE__);
		} else
			set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not access shared buffer",__FILE__,__FUNCTION__,__LINE__);
		
		if (wait_after_data)
			usleep(WAIT_PERIOD_US);
		
		//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Repeat loop",__FILE__,__FUNCTION__,__LINE__);
	}
	
	log_message(RL_DEBUG,"%s:%s(%d):Thread exited main loop",__FILE__,__FUNCTION__,__LINE__);
	
	// TODO: Close socket, etc
	if (sockfd_listen >= 0)
		close(sockfd_listen);
	if (sockfd_receive >= 0)
		close(sockfd_receive);
	
	if (local_buf != NULL) {
		free(local_buf);
		local_buf = NULL;
	}
	
	set_thread_state(st, CS_DONE,"%s:%s(%d):Thread is done",__FILE__,__FUNCTION__,__LINE__,get_thread_type_str(st->type));
	return NULL;
}

static void * _threaded_writer(void *arg) {
	int n_sg; // number of scatter gather files
	SGPlan *sgpln; // SGPlan for reading data
	uint32_t *local_buf = NULL; // local reader buffer
	int n_frames = 0; // number of frames available in buffer
	int n_frames_copied = 0; // number of frames copied from local to shared buffer
	int n_frames_this_copy = 0; // number for frames copied per iteration
	int max_frames_in_buffer = 0;
	int wait_after_data = 0;
	
	sgcomm_thread *st = (sgcomm_thread *)arg; // general thread message
	writer_msg *msg = (writer_msg *)(st->type_msg); // type specific message
	shared_buffer *src = msg->src;
	ctrl_state ctrl;
	
	/* Set this thread just started (i.e. not yet in loop) */
	set_thread_state(st, CS_START, "%s:%s(%d):Thread started",__FILE__,__FUNCTION__,__LINE__);
	
	/* Make scatter-gather plan */
	n_sg = make_sg_write_plan(&sgpln, msg->pattern, msg->fmtstr, msg->mod_list, msg->n_mod, msg->disk_list, msg->n_disk); // REDO: n_sg = 1;
	if (n_sg <= 0)
		set_thread_state(st, CS_ERROR, "%s:%s(%d):SGPlan failed, returned %d",__FILE__,__FUNCTION__,__LINE__,n_sg);
	
	log_message(RL_INFO,"%s:%s(%d):Write-mode SGPlan created from %d files",__FILE__,__FUNCTION__,__LINE__,n_sg);
	
	// TODO: Need to set the local buffer to some size that is a multiple
	// of the number of frames written per scatter-gather block.
	/* THE LOCAL BUFFER IS ASSUMED LARGER THAN THE SHARED BUFFER */
	if (obtain_data_lock(src) == 0) {
		src->frame_size = 264; // REDO: We need some other way to set the frame sizes
		max_frames_in_buffer = (uint32_t)src->buf_size/src->frame_size;
		local_buf = (uint32_t *)malloc(max_frames_in_buffer*src->frame_size*sizeof(uint32_t));
		if (release_data_lock(src) != 0)
			set_thread_state(st, CS_ERROR, "%s:%s(%d):Cannot release shared buffer",__FILE__,__FUNCTION__,__LINE__);
	} else
		set_thread_state(st, CS_ERROR, "%s:%s(%d):Cannot access shared buffer",__FILE__,__FUNCTION__,__LINE__);
	
	log_message(RL_DEBUG,"%s:%s(%d):Local buffer created",__FILE__,__FUNCTION__,__LINE__);
	
	/* Set this thread entering infinite loop section */
	if (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP))
		set_thread_state(st, CS_RUN, NULL);
	
	log_message(RL_DEBUG,"%s:%s(%d):Thread set to enter loop",__FILE__,__FUNCTION__,__LINE__);
	
	/* Continue working until stop condition */
	while (get_thread_state(st, &ctrl) == 0 && !(ctrl >= CS_STOP)) {
		/* If this thread is in a wait state, sleep and repeat check */
		if (ctrl == CS_WAIT) {
			usleep(WAIT_PERIOD_US);
			continue;
		}
		
		/* If local buffer is empty, copy from shared buffer. */
		if (n_frames_copied < max_frames_in_buffer) {
			//~ log_message(RL_DEBUGVVV,"%s:%s(%d):                                             Try to get data lock...",__FILE__,__FUNCTION__,__LINE__);
			if (obtain_data_lock(src) == 0) {
				/* Only copy if shared buffer is not empty */
				if (src->n_frames > 0) {
					/* Calculate copy size, all frames if possible,
					 * otherwise as many as the shared data buffer can keep */
					n_frames_this_copy = src->n_frames < (max_frames_in_buffer - n_frames_copied) ? 
						(src->n_frames) : (max_frames_in_buffer-n_frames_copied);
					
					log_message(RL_DEBUGVVV,"%s:%s(%d):Copying %u/%u frames (buffer space left %u)",__FILE__,__FUNCTION__,__LINE__,n_frames_this_copy,(src->n_frames),(max_frames_in_buffer-n_frames_copied));
					
					/* Copy data from shared buffer and reset frame 
					 * count. */
					memcpy((void *)(local_buf+n_frames_copied*(src->frame_size)),src->buf,
							n_frames_this_copy*src->frame_size*sizeof(uint32_t));
					
					log_message(RL_DEBUGVVV,"%s:%s(%d):Copied %u frames of data [%u .. %u] from shared to local buffer",__FILE__,__FUNCTION__,__LINE__,n_frames_this_copy,src->buf[0],src->buf[n_frames_this_copy*src->frame_size-1]);
					
					n_frames_copied += n_frames_this_copy;
					if (src->n_frames > n_frames_this_copy) {
						log_message(RL_DEBUGVVV,"%s:%s(%d):Moved %u frames of data in shared buffer",__FILE__,__FUNCTION__,__LINE__,(src->n_frames-n_frames_this_copy));
						memmove(src->buf,src->buf+n_frames_this_copy*src->frame_size,(src->n_frames-n_frames_this_copy)*src->frame_size*sizeof(uint32_t));
					}
					src->n_frames -= n_frames_this_copy;
					
					//~ if (src->n_frames > 0)
						//~ log_message(RL_DEBUGVVV,"%s:%s(%d):%u frames of data [%u .. %u] left in shared buffer",__FILE__,__FUNCTION__,__LINE__,src->n_frames,src->buf[0],src->buf[(src->n_frames)*(src->frame_size)-1]);
					//~ else
						//~ log_message(RL_DEBUGVVV,"%s:%s(%d):%u frames of data left in shared buffer",__FILE__,__FUNCTION__,__LINE__,src->n_frames);
					
					/* If shared buffer was empty, optimistic that we don't 
					 * have to wait on next iteration */
					wait_after_data = 0;
				} else {
					wait_after_data = 1;
					//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Shared buffer empty, set wait state",__FILE__,__FUNCTION__,__LINE__);
				}
				
				/* Finally, release lock on shared data */
				if (release_data_lock(src) != 0) {
					set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not release shared buffer",__FILE__,__FUNCTION__,__LINE__);
					break;
				}
				//~ log_message(RL_DEBUGVVV,"%s:%s(%d):                                         ...released data lock",__FILE__,__FUNCTION__,__LINE__);
				if (wait_after_data)
					usleep(WAIT_PERIOD_US);
			} else
				set_thread_state(st, CS_ERROR, "%s:%s(%d):Could not access shared buffer",__FILE__,__FUNCTION__,__LINE__);
			
		} else {
			/* If the local buffer is full, write to disk */
			log_message(RL_DEBUG,"%s:%s(%d):Local buffer full, writing to disk",__FILE__,__FUNCTION__,__LINE__);
			// TODO: Scatter-gather write
			write_vdif_frames(sgpln, local_buf, n_frames_copied);
			log_message(RL_DEBUG,"%s:%s(%d):Writing to disk done, next",__FILE__,__FUNCTION__,__LINE__);
			n_frames_copied = 0;
		}
			
		// TODO: Add other writer main loop tasks here
		
		//~ log_message(RL_DEBUGVVV,"%s:%s(%d):Repeat loop",__FILE__,__FUNCTION__,__LINE__);
	}
	
	// TODO: When stopped, do a check if there is data to write out (also check for other threads)
	if (n_frames_copied > 0) {
		log_message(RL_INFO,"%s:%s(%d):Found %u unwritten frames, writing to disk",__FILE__,__FUNCTION__,__LINE__,n_frames_copied);
		write_vdif_frames(sgpln, local_buf, n_frames_copied);
		n_frames_copied = 0;
	}
	
	log_message(RL_DEBUG,"%s:%s(%d):Thread exited main loop",__FILE__,__FUNCTION__,__LINE__);
	
	/* In case we exited without properly handling this memory, as is 
	 * the case when frame request returned non-positive number. */
	if (local_buf != NULL) {
		free(local_buf);
		local_buf = NULL;
	}
	
	close_sg_write_plan(sgpln);
	
	set_thread_state(st, CS_DONE,"%s:%s(%d):Thread is done",__FILE__,__FUNCTION__,__LINE__);
	return NULL;
}

static void * _threaded_controller(void *arg) {
	//~ const uint8_t pttype = TT_CONTROLLER;
	//~ pt_controller_msg *msg = (pt_controller_msg *)arg; // in message
	//~ 
	//~ set_thread_state(pttype, CS_START);
	//~ syslog(LOG_INFO,"%s:%s thread started (%lu)",__FUNCTION__,get_thread_type_str(pttype),pthread_self());
	//~ 
	//~ // TODO: Fill controller thread tasks
	//~ 
	//~ set_thread_state(pttype, CS_DONE);
	//~ syslog(LOG_INFO,"%s:%s thread stopped (%lu)",__FUNCTION__,get_thread_type_str(pttype),pthread_self());
	return NULL;
}

static int _get_status_reader(status_update *su) {
	return 0;
}
static int _get_status_transmitter(status_update *su) {
	return 0;
}
static int _get_status_receiver(status_update *su) {
	return 0;
}
static int _get_status_writer(status_update *su) {
	return 0;
}
