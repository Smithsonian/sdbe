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

#include "sgcomm_report.h"
#include "sgcomm_net.h"
#include "sgcomm_threads.h"

#define MAIN_WAIT_PERIOD_US 500000
#define SHARED_BUFFER_SIZE_TX (2056*10*32)
#define SHARED_BUFFER_SIZE_RX (2056*10*32)

/* Master thread */
sgcomm_thread st_main = {
	.type = TT_MAIN,
	.state = CS_INIT,
	.type_msg = NULL
};

/* Event handlers */
void handle_sigint(int signum);
void handle_exit(void);

/* Misc */
void print_rusage(const char *tag,struct rusage *ru);

int main(int argc, char **argv) {
	/* Slave threads */
	sgcomm_thread *st_rd; // Reader
	sgcomm_thread *st_tx; // Transmitter
	sgcomm_thread *st_rx; // Receiver
	sgcomm_thread *st_wr; // Writer
	shared_buffer *sbtx; // shared buffer for read+transmit
	shared_buffer *sbrx; // shared buffer for receive+write
	
	/* Reader message parameters */
	char *fmtstr = "/mnt/disks/%u/%u/data/%s";
	char *pattern_read = "input.vdif";
	char *pattern_write = "output2.vdif";
	int n_mod = 4;
	int mod_list[4] = { 1, 2, 3, 4};
	int n_disk = 8;
	int disk_list_read[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	int disk_list_write[8] = { 1, 0, 2, 3, 4, 5, 6, 7 };
	
	/* Transmitter message parameters */
	char *host = "localhost";
	uint16_t port;
	if (argc > 1)
		port = atoi(argv[1]);
	else
		port = 61234;
	
	/* This thread */
	sgcomm_thread *st = &st_main;
	ctrl_state state;
	
	log_message(RL_DEBUG,"%s:Creating shared buffer",__FUNCTION__);
	
	/* Initialize shared data buffer */
	sbtx = create_shared_buffer(SHARED_BUFFER_SIZE_TX);
	if (sbtx == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create shared buffer for read+transmit",__FUNCTION__,__LINE__);
	sbrx = create_shared_buffer(SHARED_BUFFER_SIZE_RX);
	if (sbrx == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create shared buffer for receive+write",__FUNCTION__,__LINE__);
	
	log_message(RL_DEBUG,"%s:Creating slave threads",__FUNCTION__);
	
	/* Create thread instances */
	st_rd = create_thread(TT_READER);
	if (st_rd == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create reader thread",__FUNCTION__,__LINE__);
	st_tx = create_thread(TT_TRANSMITTER);
	if (st_tx == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create transmitter thread",__FUNCTION__,__LINE__);
	st_rx = create_thread(TT_RECEIVER);
	if (st_rx == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create receiver thread",__FUNCTION__,__LINE__);
	st_wr = create_thread(TT_WRITER);
	if (st_wr == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create writer thread",__FUNCTION__,__LINE__);
	
	log_message(RL_DEBUG,"%s:Initializing thread messages",__FUNCTION__);
	
	/* Initialize thread messages */
	init_reader_msg((reader_msg *)st_rd->type_msg, sbtx, pattern_read, fmtstr,
					mod_list, n_mod, disk_list_read, n_disk);
	init_transmitter_msg((transmitter_msg *)st_tx->type_msg, sbtx,
					host, port);
	init_receiver_msg((receiver_msg *)st_rx->type_msg, sbrx,
					host, port);
	init_writer_msg((writer_msg *)st_wr->type_msg, sbrx, pattern_write, fmtstr,
					mod_list, n_mod, disk_list_write, n_disk);
	
	//~ exit(EXIT_SUCCESS);
	
	/* Start receiver thread */
	if (start_thread(st_rx) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot start receiver thread",__FUNCTION__,__LINE__);
	/* Pause, then see if receiver has error, if so, abort */
	usleep(MAIN_WAIT_PERIOD_US);
	if ((get_thread_state(st_rx,&state) == 0) && (state >= CS_STOP)) {
		set_thread_state(st,CS_ERROR,"%s(%d):Receiver terminated prematurely, aborting start.",__FUNCTION__,__LINE__);
	} else {
		if (start_thread(st_tx) != 0)
			set_thread_state(st,CS_ERROR,"%s(%d):Cannot start transmitter thread",__FUNCTION__,__LINE__);
		/* Pause, then see if transmitter has error, if so, abort */
		usleep(MAIN_WAIT_PERIOD_US);
		if ((get_thread_state(st_tx,&state) == 0) && (state >= CS_STOP)) {
			set_thread_state(st,CS_ERROR,"%s(%d):Transmitter terminated prematurely, aborting start.",__FUNCTION__,__LINE__);
		} else {
			if (start_thread(st_wr) != 0)
				set_thread_state(st,CS_ERROR,"%s(%d):Cannot start writer thread",__FUNCTION__,__LINE__);
			if (start_thread(st_rd) != 0)
				set_thread_state(st,CS_ERROR,"%s(%d):Cannot start reader thread",__FUNCTION__,__LINE__);
		}
	}
	
	//~ log_message(RL_DEBUG,"%s:Entering main thread run loop",__FUNCTION__);
	
	if ((get_thread_state(st,&state) == 0) && !(state >= CS_STOP))
		set_thread_state(st,CS_RUN,"%s:Thread running",__FUNCTION__);
	while ((get_thread_state(st,&state) == 0) && !(state >= CS_STOP)) {
				
		// TODO: do something
		
		usleep(MAIN_WAIT_PERIOD_US);
		
		/* If any thread has a problem, stop all of them */
		if ( ((get_thread_state(st_rd,&state) == 0) && (state >= CS_ERROR)) ||
			 ((get_thread_state(st_tx,&state) == 0) && (state >= CS_ERROR)) ||
			 ((get_thread_state(st_rx,&state) == 0) && (state >= CS_ERROR)) ||
			 ((get_thread_state(st_wr,&state) == 0) && (state >= CS_ERROR)) ) {
			// TODO: Some cleanup?
			break;
		}
		
		/* If all threads are stopped, break */
		if ( ((get_thread_state(st_rd,&state) == 0) && (state >= CS_STOP)) &&
			 ((get_thread_state(st_tx,&state) == 0) && (state >= CS_STOP)) &&
			 ((get_thread_state(st_rx,&state) == 0) && (state >= CS_STOP)) &&
			 ((get_thread_state(st_wr,&state) == 0) && (state >= CS_STOP)) ) {
			log_message(RL_NOTICE,"%s:All threads stopped of their own volition",__FUNCTION__);
			break;
		}
		
		/* If reader thread AND receiver are done, stop transmitter */
		if ( (get_thread_state(st_rd,&state) == 0) && (state >= CS_STOP) && 
			 (get_thread_state(st_rx,&state) == 0) && (state >= CS_STOP) && 
			 (get_thread_state(st_tx,&state) == 0) && (state < CS_STOP)) {
			log_message(RL_NOTICE,"%s:Reader and receiver are done, stop transmitter",__FUNCTION__);
			/* Two wait periods should be enough - reader is the only
			 * other thread that can cause transmitter to wait on a 
			 * resource, and then it will only be a single wait. */
			usleep(MAIN_WAIT_PERIOD_US);
			usleep(MAIN_WAIT_PERIOD_US);
			if (stop_thread(st_tx) != 0)
				set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop transmitter thread",__FUNCTION__,__LINE__);
		}
		
		/* If receiver thread is done, stop writer */
		if ( (get_thread_state(st_rx,&state) == 0) && (state >= CS_STOP) && 
			 (get_thread_state(st_wr,&state) == 0) && (state < CS_STOP)) {
			log_message(RL_NOTICE,"%s:Receiver is done, stop writer",__FUNCTION__);
			/* Two wait periods should be enough - receiver is the only
			 * other thread that can cause writer to wait on a resource,
			 * and then it will only be a single wait. */
			usleep(MAIN_WAIT_PERIOD_US);
			usleep(MAIN_WAIT_PERIOD_US);
			if (stop_thread(st_wr) != 0)
				set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop writer thread",__FUNCTION__,__LINE__);
		}
	}
	
	log_message(RL_DEBUG,"%s:Stopping slave threads",__FUNCTION__);
	
	/* Stop slave threads, first tx side */
	if ( (get_thread_state(st_rd,&state) == 0) && (state < CS_STOP) && (state > CS_INIT) && stop_thread(st_rd) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop reader thread",__FUNCTION__,__LINE__);
	log_message(RL_DEBUGVVV,"%s:Reader thread stopped",__FUNCTION__);
	if ( (get_thread_state(st_tx,&state) == 0) && (state < CS_STOP) && (state > CS_INIT) && stop_thread(st_tx) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop transmitter thread",__FUNCTION__,__LINE__);
	log_message(RL_DEBUGVVV,"%s:Transmitter thread stopped",__FUNCTION__);
	/* then rx side */
	
	sleep(1);
	
	if ( (get_thread_state(st_rx,&state) == 0) && (state < CS_STOP) && (state > CS_INIT) && stop_thread(st_rx) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop receiver thread",__FUNCTION__,__LINE__);
	log_message(RL_DEBUGVVV,"%s:Receiver thread stopped",__FUNCTION__);
	if ( (get_thread_state(st_wr,&state) == 0) && (state < CS_STOP) && (state > CS_INIT) && stop_thread(st_wr) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop writer thread",__FUNCTION__,__LINE__);
	log_message(RL_DEBUGVVV,"%s:Writer thread stopped",__FUNCTION__);
	
	log_message(RL_DEBUG,"%s:Destroying shared buffer",__FUNCTION__);
	
	/* Destroy shared data buffer */
	if (destroy_shared_buffer(&sbtx) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot destroy shared buffer for read+transmit",__FUNCTION__,__LINE__);
	if (destroy_shared_buffer(&sbrx) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot destroy shared buffer for receive+write",__FUNCTION__,__LINE__);
	
	log_message(RL_DEBUG,"%s:Destroying slave threads",__FUNCTION__);
	
	/* Destroy threads */
	destroy_thread(&st_rd);
	destroy_thread(&st_tx);
	destroy_thread(&st_rx);
	destroy_thread(&st_wr);
	
	log_message(RL_DEBUG,"%s:Everything is done, goodbye",__FUNCTION__);
	
	/* That's all folks! */
	// TODO: Report that we're done
	return EXIT_SUCCESS;
}

void handle_exit(void) {
	syslog(LOG_INFO,"%s:At exit..",__FUNCTION__);
	struct rusage ru;
	if (getrusage(RUSAGE_SELF,&ru) != 0)
		perror("getrusage");
	else 
		print_rusage("",&ru);
}

void print_rusage(const char *tag,struct rusage *ru) {
	printf("%sResoures usage statistics:\n",tag);
	printf("%s  Max res size (KiB):              %ld\n",tag,ru->ru_maxrss);
	printf("\n");
}

void handle_sigint(int signum) {
	log_message(RL_NOTICE,"%s:SIGINT received, stopping %s thread",__FUNCTION__,get_thread_type_str(st_main.type));
	/* Cannot stop_thread on main, since it's not pthread-ed */
	//~ stop_thread(&st_main);
	set_thread_state(&st_main, CS_STOP, NULL);
}

static __attribute__((constructor)) void initialize() {
	/* Start logging */
	open_logging(RL_DEBUGVVV,RLT_STDOUT,NULL);
	//~ open_logging(RL_DEBUG,RLT_STDOUT,NULL);
	//~ open_logging(RL_INFO,RLT_STDOUT,NULL);
	//~ open_logging(RL_NOTICE,RLT_STDOUT,NULL);
	
	/* Set signal handlers */
	if (signal(SIGINT, handle_sigint) == SIG_IGN)
		signal(SIGINT, SIG_IGN);
	
	/* Register exit method */
	atexit(&handle_exit);
}

static __attribute__((destructor)) void deinitialize() {
	/* End logging */
	close_logging();
}
