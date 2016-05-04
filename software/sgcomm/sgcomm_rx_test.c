#include <error.h>
#include <execinfo.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>

#include <netinet/in.h>
#include <netdb.h> 

#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h> 

#include "sgcomm_report.h"
#include "sgcomm_net.h"
#include "sgcomm_threads.h"

#define MAIN_WAIT_PERIOD_US 500000
#define SHARED_BUFFER_SIZE_TX (530*8192*32)
#define SHARED_BUFFER_SIZE_RX (2056*1215*32)

#define BT_BUFFER_SIZE 0x100
#define BT_FILENAME "crashreport.bt"
#define MAX_LEN_STR_TIMESTAMP 0x100
#define MAX_LEN_STR_HDR 0x400
#define STR_FMT_TIMESTAMP_LONG "%Y-%m-%d %H:%M:%S"

/* Master thread */
sgcomm_thread st_main = {
	.type = TT_MAIN,
	.state = CS_INIT,
	.type_msg = NULL
};

/* Event handlers */
void handle_sigint(int signum);
void handle_exit(void);

/* Arguments */
// default values
char *fmtstr = "/mnt/disks/%u/%u/data/%s";
char *pattern_write = "output.vdif";
char *host = "localhost";
uint16_t port = 61234;
int n_mod = 4;
int mod_list[4] = { 1, 2, 3, 4};
int n_disk = 8;
int disk_list_write[8] = { 1, 0, 2, 3, 4, 5, 6, 7 };
char *log_filename = NULL;
// parse from command line
void parse_arguments(int argc, char **argv);
void print_usage(const char *cmd);
void print_arguments(void);
int csv_to_int(char *str, int **list); 

/* Misc */
void print_rusage(const char *tag,struct rusage *ru);

// Signal handlers
void handle_sigfpe(int sig);
void handle_sigsegv(int sig);
// backtrace utility
static void print_backtrace_to_file(const char *filename, const char *hdr);
// get nice time string
static void get_timestamp_long(char *time_str);

int main(int argc, char **argv) {
	/* Slave threads */
	sgcomm_thread *st_rx; // Receiver
	sgcomm_thread *st_wr; // Writer
	shared_buffer *sbrx; // shared buffer for receive+write
	
	// set up according to input arguments
	parse_arguments(argc,argv);

	/* Start logging */
	if (log_filename == NULL) {
		open_logging(RL_DEBUGVVV,RLT_STDOUT,NULL);
	} else {
		open_logging(RL_DEBUGVVV,RLT_FILE,log_filename);
	}

	
	log_message(RL_NOTICE,"%s:Using output file '%s' matching pattern '%s'",__FUNCTION__,pattern_write,fmtstr);
	log_message(RL_NOTICE,"%s:Receiving on %s:%u",__FUNCTION__,host,port);
	
	/* This thread */
	sgcomm_thread *st = &st_main;
	ctrl_state state;
	
	log_message(RL_DEBUG,"%s:Creating shared buffer",__FUNCTION__);
	
	/* Initialize shared data buffer */
	sbrx = create_shared_buffer(SHARED_BUFFER_SIZE_RX);
	if (sbrx == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create shared buffer for receive+write",__FUNCTION__,__LINE__);
	
	log_message(RL_DEBUG,"%s:Creating slave threads",__FUNCTION__);
	
	/* Create thread instances */
	st_rx = create_thread(TT_RECEIVER);
	if (st_rx == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create receiver thread",__FUNCTION__,__LINE__);
	st_wr = create_thread(TT_WRITER);
	if (st_wr == NULL)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot create writer thread",__FUNCTION__,__LINE__);
	
	log_message(RL_DEBUG,"%s:Initializing thread messages",__FUNCTION__);
	
	/* Initialize thread messages */
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
		if (start_thread(st_wr) != 0)
			set_thread_state(st,CS_ERROR,"%s(%d):Cannot start writer thread",__FUNCTION__,__LINE__);
	}
	
	//~ log_message(RL_DEBUG,"%s:Entering main thread run loop",__FUNCTION__);
	
	if ((get_thread_state(st,&state) == 0) && !(state >= CS_STOP))
		set_thread_state(st,CS_RUN,"%s:Thread running",__FUNCTION__);
	while ((get_thread_state(st,&state) == 0) && !(state >= CS_STOP)) {
				
		// TODO: do something
		
		usleep(MAIN_WAIT_PERIOD_US);
		
		/* If any thread has a problem, stop all of them */
		if ( ((get_thread_state(st_rx,&state) == 0) && (state >= CS_ERROR)) ||
			 ((get_thread_state(st_wr,&state) == 0) && (state >= CS_ERROR)) ) {
			// TODO: Some cleanup?
			break;
		}
		
		/* If all threads are stopped, break */
		if ( ((get_thread_state(st_rx,&state) == 0) && (state >= CS_STOP)) &&
			 ((get_thread_state(st_wr,&state) == 0) && (state >= CS_STOP)) ) {
			log_message(RL_NOTICE,"%s:All threads stopped of their own volition",__FUNCTION__);
			break;
		}
		
		/* If receiver thread is done, stop writer */
		if ( (get_thread_state(st_rx,&state) == 0) && (state == CS_DONE) && 
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
	
	if ( (get_thread_state(st_rx,&state) == 0) && (state < CS_STOP) && (state > CS_INIT) && stop_thread(st_rx) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop receiver thread",__FUNCTION__,__LINE__);
	log_message(RL_DEBUGVVV,"%s:Receiver thread stopped",__FUNCTION__);
	if ( (get_thread_state(st_wr,&state) == 0) && (state < CS_STOP) && (state > CS_INIT) && stop_thread(st_wr) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot stop writer thread",__FUNCTION__,__LINE__);
	log_message(RL_DEBUGVVV,"%s:Writer thread stopped",__FUNCTION__);
	
	log_message(RL_DEBUG,"%s:Destroying shared buffer",__FUNCTION__);
	
	/* Destroy shared data buffer */
	if (destroy_shared_buffer(&sbrx) != 0)
		set_thread_state(st,CS_ERROR,"%s(%d):Cannot destroy shared buffer for receive+write",__FUNCTION__,__LINE__);
	
	log_message(RL_DEBUG,"%s:Destroying slave threads",__FUNCTION__);
	
	/* Destroy threads */
	destroy_thread(&st_rx);
	destroy_thread(&st_wr);
	
	log_message(RL_DEBUG,"%s:Everything is done, goodbye",__FUNCTION__);
	
	/* That's all folks! */
	// TODO: Report that we're done
	return EXIT_SUCCESS;
}

void parse_arguments(int argc, char **argv) {
	int c;
	int *tmp_list;
	int ii;
	
	while (1)
	{
		static struct option long_options[] =
		{
			{"address"              , required_argument, 0, 'a'},
			{"disk-list"            , required_argument, 0, 'd'},
			{"output-format-string" , required_argument, 0, 'f'},
			{"help"                 , no_argument,       0, 'h'},
			{"logfile"              , required_argument, 0, 'l'},
			{"module-list"          , required_argument, 0, 'm'},
			{"output"               , required_argument, 0, 'o'},
			{"port"                 , required_argument, 0, 'p'},
			{0, 0, 0, 0}
		};
		/* getopt_long stores the option index here. */
		int option_index = 0;
		
		c = getopt_long(argc, argv, "a:d:f:hl:m:o:p:",
							long_options, &option_index);
	
		/* Detect the end of the options. */
		if (c == -1)
		{
			break;
		}
		
		switch (c)
		{
			case 'a':
				host = optarg;
				break;
			case 'd':
				n_disk = csv_to_int(optarg, &tmp_list);
				for (ii=0; ii<n_disk; ii++) {
					disk_list_write[ii] = tmp_list[ii];
				}
				free(tmp_list);
				break;
			case 'f':
				fmtstr = optarg;
				break;
			case 'h':
				print_usage(argv[0]);
				exit(EXIT_SUCCESS);
				break;
			case 'l':
				log_filename = optarg;
				break;
			case 'm':
				n_mod = csv_to_int(optarg, &tmp_list);
				for (ii=0; ii<n_mod; ii++) {
					mod_list[ii] = tmp_list[ii];
				}
				free(tmp_list);
				break;
			case 'o':
				pattern_write = optarg;
				break;
			case 'p':
				port = (uint16_t)atoi(optarg);
				break;
			case '?':
				/* getopt_long already printed an error message. */
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}
	print_arguments();
}

void print_usage(const char *cmd) {
	printf(
		"Usage: %s [OPTIONS]\n"
		"Receive VDIF packets on network socket and write to scatter-gather.\n"
		"\n"
		"Mandatory arguments to long options are mandatory to short options too.\n"
		"-a, --address=ADDR                  network address at which to receive packets\n"
		"-d, --disk-list=LIST                comma-separated list of disks to write to\n"
		"-f, --output-format-string=FMTSTR   format string to convert mod-/disk-list and\n"
		"                                    output filename to full path\n"
		"-h, --help                          display this message\n"
		"-l, --log=LOGFILE                   log to given filename\n"
		"-m, --module-list=LIST              comma-separated list of modules to write to\n"
		"-o, --output                        output filename\n"
		"-p, --port                          network port on which to receive packets\n"
		"\n"
		"The full path of output files are built using a printf-style substitution of\n"
		"each module, each disk, and the output filename in the format string, which\n"
		"should contain two unsigned integer (%%u) and one string (%%s) substitutions,\n"
		"in that order. For example, given the format string '/mnt/disks/%%u/%%u/data/%%s',\n"
		"disk-list {0,1}, module-list {3,4}, and output filename 'test.vdif', the output\n"
		"files created are:\n"
		"\t/mnt/disks/3/0/data/test.vdif\n"
		"\t/mnt/disks/3/1/data/test.vdif\n"
		"\t/mnt/disks/4/0/data/test.vdif\n"
		"\t/mnt/disks/4/1/data/test.vdif\n"
		"\n"
		,cmd);
}

void print_arguments(void) {
	char mod_list_str[256];
	char disk_list_str[256];
	int ii;
	int cidx;
	cidx = 0;
	for (ii=0; ii<n_mod; ii++) {
		if (ii>0) {
			sprintf(mod_list_str+cidx,",");
			cidx++;
		}
		sprintf(mod_list_str+cidx,"%2d",mod_list[ii]);
		cidx += 2;
	}
	cidx = 0;
	for (ii=0; ii<n_disk; ii++) {
		if (ii>0) {
			sprintf(disk_list_str+cidx,",");
			cidx++;
		}
		sprintf(disk_list_str+cidx,"%2d",disk_list_write[ii]);
		cidx += 2;
	}
	printf(
		"Called with:\n"
		"\tfmtstr=%s\n"
		"\tpattern_write=%s\n"
		"\thost=%s\n"
		"\tport=%d\n"
		"\tmod_list=%s\n"
		"\tdisk_list=%s\n"
		"\tlog_filename=%s\n"
		,fmtstr,pattern_write,host,(int)port,mod_list_str,disk_list_str,
		log_filename==NULL ? "NULL" : log_filename);
}

int csv_to_int(char *str, int **list) {
	char *num_str;
	int ii = 0;
	*list = (int *)malloc(100*sizeof(int));
	num_str = strtok(str, ",");
	while (num_str != NULL) {
		(*list)[ii++] = atoi(num_str);
		num_str = strtok(NULL,",");
	}
	return ii;
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

void handle_sigfpe(int sig) {
	fprintf(stderr,"SIGFPE received\n");

	char tsstr[MAX_LEN_STR_TIMESTAMP];
	char hdr[MAX_LEN_STR_HDR];
	get_timestamp_long(tsstr);
	snprintf(hdr, MAX_LEN_STR_HDR, "SIGFPE received at %s:\n", tsstr);
	print_backtrace_to_file(BT_FILENAME, hdr);

	exit(EXIT_FAILURE);
}

void handle_sigsegv(int sig) {
	fprintf(stderr,"SIGSEGV received\n");

	char tsstr[MAX_LEN_STR_TIMESTAMP];
	char hdr[MAX_LEN_STR_HDR];
	get_timestamp_long(tsstr);
	snprintf(hdr, MAX_LEN_STR_HDR, "SIGSEGV received at %s:\n", tsstr);
	print_backtrace_to_file(BT_FILENAME, hdr);

	exit(EXIT_FAILURE);
}

static void print_backtrace_to_file(const char *filename, const char *hdr) {
	int fd;
	void *buffer[BT_BUFFER_SIZE];
	int n_sym;
	fd = open(filename,O_WRONLY|O_APPEND|O_CREAT,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH);
	if (fd == -1) {
		perror("open");
		return;
	}
	// write header
	write(fd, hdr, strlen(hdr));
	n_sym = backtrace(buffer,BT_BUFFER_SIZE);
	backtrace_symbols_fd(buffer, n_sym, fd);
	if (close(fd) == -1) {
		fprintf(stderr,"WARNING, backtrace logging file '%s' may not be closed properly\n",filename);
		perror("close");
	}
}
static void get_timestamp_long(char *time_str) {
	time_t t;
	struct tm timestamp;
	
	time(&t);
	localtime_r(&t,&timestamp);
	strftime(time_str,MAX_LEN_STR_TIMESTAMP,STR_FMT_TIMESTAMP_LONG,&timestamp);
}

struct sigaction sa_sigfpe = {
	.sa_handler = handle_sigfpe,
	.sa_mask = 0,
	.sa_flags = 0
};

struct sigaction sa_sigsegv = {
	.sa_handler = handle_sigsegv,
	.sa_mask = 0,
	.sa_flags = 0
};

static __attribute__((constructor)) void initialize() {
//	/* Start logging */
//	if (log_filename == NULL) {
//		open_logging(RL_DEBUGVVV,RLT_STDOUT,NULL);
//	} else {
//		open_logging(RL_DEBUGVVV,RLT_FILE,log_filename);
//	}
	//~ open_logging(RL_DEBUG,RLT_STDOUT,NULL);
	//~ open_logging(RL_INFO,RLT_STDOUT,NULL);
	//~ open_logging(RL_NOTICE,RLT_STDOUT,NULL);
	
	/* Set signal handlers */
	if (signal(SIGINT, handle_sigint) == SIG_IGN)
		signal(SIGINT, SIG_IGN);
	sigaction(SIGFPE, &sa_sigfpe, NULL);
	sigaction(SIGSEGV, &sa_sigsegv, NULL);
	
	/* Register exit method */
	atexit(&handle_exit);
}

static __attribute__((destructor)) void deinitialize() {
	/* End logging */
	close_logging();
}
