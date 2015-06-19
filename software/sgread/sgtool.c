#include <getopt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <sys/time.h>
#include <sys/resource.h>

#include "sg_access.h"
#include "vdif_ext.h"

#include "print_rusage.h"

#define DEBUG_LEVEL_DEBUG 40
#define DEBUG_LEVEL_INFO 30
#define DEBUG_LEVEL_WARNING 20
#define DEBUG_LEVEL_ERROR 10
//~ #define DEBUG_LEVEL DEBUG_LEVEL_DEBUG
//~ #define DEBUG_LEVEL DEBUG_LEVEL_INFO
//~ #define DEBUG_LEVEL DEBUG_LEVEL_WARNING
//~ #define DEBUG_LEVEL DEBUG_LEVEL_ERROR


#define MAX_FILENAME_LENGTH 0x100

#define PATH_FORMAT_STRING "/mnt/disks/%u/%u/data/%s"

// debugging messages
#ifdef DEBUG_LEVEL
	void debug_msg(const char *msg, const char *filename, int linenum);
	#define DEBUGMSG(m) debug_msg(m,__FILE__,__LINE__)
	void error_msg(const char *msg, const char *filename, int linenum);
	#define ERRORMSG(m) error_msg(m,__FILE__,__LINE__)
	void warning_msg(const char *msg, const char *filename, int linenum);
	#define WARNINGMSG(m) warning_msg(m,__FILE__,__LINE__)
	void info_msg(const char *msg, const char *filename, int linenum);
	#define INFOMSG(m) info_msg(m,__FILE__,__LINE__)
#endif

/* Structures for SG thread communications. */
typedef struct msg_to_sgthread { // message TO an SG thread
	SGInfo *sgi_ptr; // points to SGInfo 
	off_t iblock; // number block to read from
} MsgToSGThread;
typedef struct msg_from_sgthread { // message FROM an SG thread
	uint32_t *data_buf; // pointer to VDIF data buffer
	size_t num_frames; // number of frames in buffer
} MsgFromSGThread;

// Forward declarations
// Basic program utilities.
void initialize(void);
void parse_options(int argc, char **argv);
void print_usage(void);
void do_exit(void);
// SGInfo utilities
int mod_disk_id(int mod, int disk);
int compare_sgi(const void *a, const void *b);
int fill_sgi(SGInfo **sgi, const char *pattern, int *mod_list, int n_mod, int *disk_list, int n_disk);
// Reading VDIF from SG files
int read_block_vdif_frames(SGInfo *sgi, int n_sgi, off_t iblock, uint32_t **vdif_buf);

// Thead methods
static void * sgthread_read_block(void *arg);
static void * sgthread_fill_sgi(void *arg);

// Memory management
void free_msg_from_thread(MsgFromSGThread *msg);
void free_sg_info(SGInfo *sgi);

// Global variables
// misc control
int verbose_output = 0; // verbose output
// output control
FILE *fh_vdif = NULL; // write VDIF to file
FILE *fh_stats = NULL; // write statistics to file
char *filename_stats = NULL;
// input
char *file_pattern = NULL;
// data
uint32_t vdif_payload_size = 0;

int main(int argc, char **argv)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Start.");
	#endif

	SGInfo *sgi;
	uint32_t *vdif_buf = NULL;
	uint32_t vdif_buf_size = 0;
	
	int n_mod = 4;
	int mod_list[4] = { 1, 2, 3, 4 };
	int n_disk = 8;
	int disk_list[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	
	/* Just some initialization.
	 */
	initialize();
	
	/* Parse input arguments and set global variables accordingly.
	 */
	parse_options(argc, argv);
	
	/* Initialize the array of SGInfo structures by accessing each of 
	 * files that matches the specified input filename pattern. Also do
	 * other household tasks, like:
	 *   # Set the VDIF data payload size
	 */
	int ii, n_sgi;
	int progress = 0;
	printf("Filling SGInfos\n");
	n_sgi = fill_sgi(&sgi, file_pattern, mod_list, n_mod, disk_list, n_disk);
	printf("Found %d valid SG files.\n",n_sgi);
	//int n_sgi = read_block_vdif_frames(&sgi,&vdif_buf);
	if (n_sgi > 0)
	{
		uint32_t min_block = 0xFFFFFFFF;
		int sgi_count;
                for (sgi_count=0; sgi_count<n_sgi; sgi_count++)
		{
			if (sgi[sgi_count].sg_total_blks < min_block)
			{
				min_block = sgi[sgi_count].sg_total_blks;
			}
		}
		min_block--; // in case short block
		printf("Found at least %u blocks per file.\n",min_block);
		printf("Total read size is %lu = %ub x %uP/b x %uB/P x %dd.\n",(unsigned long int)min_block*sgi[0].sg_wr_pkts*sgi[0].pkt_size*n_sgi, min_block, sgi[0].sg_wr_pkts, sgi[0].pkt_size, n_sgi);
		uint32_t total_frames = 0;
		uint32_t block_count;
		fprintf(stdout,"Progress: %3d%%",progress);
		fflush(stdout);
		for (block_count = 0; block_count<min_block; block_count++)
		{
			total_frames += read_block_vdif_frames(sgi,n_sgi,block_count,&vdif_buf);
			if ((100*block_count)/min_block > progress)
			{
				progress = (100*block_count)/min_block;
				fprintf(stdout,"%c%c%c%c%3d%%",(char)8,(char)8,(char)8,(char)8,progress);
				fflush(stdout);
			}
			free(vdif_buf);
		}
		fprintf(stdout,"%c%c%c%c%3d%%\n",(char)8,(char)8,(char)8,(char)8,100);
		printf("Read %d VDIF frames.\n",total_frames);
	}
	
	exit(EXIT_SUCCESS);
}

#ifdef DEBUG_LEVEL
	void debug_msg(const char *msg, const char *filename, int linenum)
	{
		printf("%s:%d:DEBUG:%s\n",filename,linenum,msg);
	}

	void error_msg(const char *msg, const char *filename, int linenum)
	{
		printf("%s:%d:ERROR:%s\n",filename,linenum,msg);
	}

	void warning_msg(const char *msg, const char *filename, int linenum)
	{
		printf("%s:%d:WARNING:%s\n",filename,linenum,msg);
	}

	void info_msg(const char *msg, const char *filename, int linenum)
	{
		printf("%s:%d:INFO:%s\n",filename,linenum,msg);
	}
#endif

/* Basic program utilities.
 */
void initialize(void)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter initialize.");
	#endif
	atexit(do_exit);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave initialize.");
	#endif
}

void parse_options(int argc, char **argv)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter parse_options.");
	#endif
	
	int c;
	
	while (1)
	{
		static struct option long_options[] =
		{
			{"help"   , no_argument,       0, 'h'},
			{"stats"  , required_argument, 0, 's'},
			{"verbose", optional_argument, 0, 'v'},
			{0, 0, 0, 0}
		};
		/* getopt_long stores the option index here. */
		int option_index = 0;
		
		c = getopt_long(argc, argv, "hs:v::",
							long_options, &option_index);
	
		/* Detect the end of the options. */
		if (c == -1)
		{
			break;
		}
		
		switch (c)
		{
			case 'h':
				print_usage();
				exit(EXIT_SUCCESS);
				break;
			case 's':
				filename_stats = optarg;
				break;
			case 'v':
				if (optarg)
				{
					verbose_output = atoi(optarg);
				}
				else
				{
					verbose_output = 1;
				}
				break;
			case '?':
				/* getopt_long already printed an error message. */
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}
	
	if (optind < argc)
	{
		file_pattern = argv[optind];
	}
	else
	{
		print_usage();
		exit(EXIT_FAILURE);
	}
	
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave parse_options.");
	#endif
}

void print_usage()
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter print_usage.");
	#endif
	printf("Usage: sgtool OPTIONS FILEPATTERN\n");
	printf("\n");
	printf("OPTIONS are:\n");
	printf("  -h,     --help         Display this message.\n");
	printf("  -v [N], --verbose [N]  Verbose output, and optionally using verbosity level of N for SGInfo structures (default is 1).\n");
	printf("\n\n");
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave print_usage.");
	#endif
}

void do_exit(void)
{
	if (fh_vdif != NULL)
	{
		fclose(fh_vdif);
	}
	if (fh_stats != NULL)
	{
		fclose(fh_stats);
	}
	
	// print resource usage
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage) == 0)
	{
		printf("\nResource usage:\n");
		printRusage("", &usage);
		//~ printf("\tru_maxrss = %lu KiB;\n",usage.ru_maxrss);
		//~ printf("\n\n");
	}
	
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Stop.");
	#endif
}

/*
 * Comparison method to sort an array of SGInfo elements.
 * Arguments:
 *   const void *a -- SGInfo by reference.
 *   const void *b -- SGInfo by reference.
 * Return:
 *   int - Returns -1 if a < b, 0 if a == b, and 1 if a > b.
 * Notes:
 *   The comparison is done by comparing the timestamp on the first VDIF
 *     frame in the file associated with *a and *b.
 */
int compare_sgi(const void *a, const void *b)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter compare_sgi.");
	#endif
	SGInfo *sgi_a = (SGInfo *)a;
	SGInfo *sgi_b = (SGInfo *)b;
	int result = sgi_a->first_secs < sgi_b->first_secs ? -1 : sgi_a->first_secs > sgi_b->first_secs;
	if (result == 0)
	{
		result = sgi_a->first_frame < sgi_b->first_frame ? -1 : sgi_a->first_frame > sgi_b->first_frame;
	}
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave compare_sgi.");
	#endif
	return result;
}

/*
 * Allocate memory and fill it with SGInfo instances.
 * Arguments:
 *   SGInfo **sgi -- Address of SGInfo pointer to allocate memory.
 *   const char *pattern -- Filename pattern to search for.
 *   int *mod_list -- Array of module numbers to use.
 *   int n_mod -- Number of modules to use.
 *   int *disk_list -- Array of disk numbers to use.
 *   int n_disk -- Number of disks to use.
 * Returns:
 *   int -- Number of SGInfo instances (SG files found mathcing pattern)
 * Notes:
 *   All names that match the pattern /mnt/disks/MOD/DISK/data/PATTERN
 *     where MOD and DISK are elements of mod_list and disk_list, 
 *     respectively, and PATTERN is the string in pattern, are given
 *     to sg_access. For each valid SG file found an SGInfo entry is 
 *     allocated in the buffer pointed to by *sgi.
 *   The SGInfo entries stored in *sgi are sorted in ascending order
 *     according to the timestamp on the first VDIF frame in each SG 
 *     file.
 */
int fill_sgi(SGInfo **sgi, const char *pattern, int *mod_list, int n_mod, int *disk_list, int n_disk)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter fill_sgi.");
	#endif
	int itmp; // just a counter
	int idisk, imod; // disk, module counters
	char filename[n_mod*n_disk][MAX_FILENAME_LENGTH]; // full filename searched for
	int ithread; // thread counter
	int thread_result; // return result for pthread methods
	pthread_t sg_threads[n_mod*n_disk]; // pthreads to do filling
	int valid_sgi = 0; // number of valid SG files found
	/* Allocate temporary buffer to store maximum possible SGInfo 
	 * instances.
	 */
	SGInfo *sgi_buf = (SGInfo *)calloc(sizeof(SGInfo),n_mod*n_disk);
	/* And allocate temporary single SGInfo. */
	SGInfo *sgi_tmp = (SGInfo *)calloc(sizeof(SGInfo),1);
	#ifdef DEBUG_LEVEL
		char _dbgmsg[0x200];
	#endif
	 
	/* Step through all modules and disks, and access files that 
	 * match the pattern.
	 */
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tLaunching threads.");
	#endif
	for (imod=0; imod<n_mod; imod++)
	{
		#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
			snprintf(_dbgmsg,0x200,"\t\tmod[%d] = %d",imod,mod_list[imod]);
			DEBUGMSG(_dbgmsg);
		#endif
		for (idisk=0; idisk<n_disk; idisk++)
		{
			#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
				snprintf(_dbgmsg,0x200,"\t\t\tdisk[%d] = %d",idisk,disk_list[idisk]);
				DEBUGMSG(_dbgmsg);
			#endif
			ithread = imod*n_disk + idisk;
			snprintf(filename[ithread],MAX_FILENAME_LENGTH,PATH_FORMAT_STRING,mod_list[imod],disk_list[idisk],pattern);
			#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
				snprintf(_dbgmsg,0x200,"\t\t\tAccessing file '%s'.",filename[ithread]);
				INFOMSG(_dbgmsg);
			#endif
			thread_result = pthread_create(&(sg_threads[ithread]), NULL, &sgthread_fill_sgi, filename[ithread]);
			if (thread_result != 0)
			{
				perror("Unable to launch thread.");
				exit(EXIT_FAILURE);
			}
		}
	}
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tJoining threads.");
	#endif
	for (imod=0; imod<n_mod; imod++)
	{
		#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
			snprintf(_dbgmsg,0x200,"\t\tmod[%d] = %d",imod,mod_list[imod]);
			DEBUGMSG(_dbgmsg);
		#endif
		for (idisk=0; idisk<n_disk; idisk++)
		{
			#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
				snprintf(_dbgmsg,0x200,"\t\t\tdisk[%d] = %d",idisk,disk_list[idisk]);
				DEBUGMSG(_dbgmsg);
			#endif
			ithread = imod*n_disk + idisk;
			thread_result = pthread_join(sg_threads[ithread],(void *)&sgi_tmp);
			if (thread_result != 0)
			{
				perror("Unable to join thread.");
				exit(EXIT_FAILURE);
			}
			if (sgi_tmp->smi.mmfd > 0)
			{
				memcpy(sgi_buf+valid_sgi++, sgi_tmp, sizeof(SGInfo));
				#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
					sg_report(&(sgi_buf[valid_sgi-1]),"\tSG report (sgi_buf):");
					DEBUGMSG("\t\t\tClosing SGInfo.");
				#endif
				sg_close(sgi_tmp);
			}
			/* Free the temporary SGInfo resources, but DO NOT free
			 * the malloc'ed NAME to which we still keep a pointer.
			 */
			free(sgi_tmp);
		}
	}
	if (valid_sgi == 0)
	{
		/* Done with this, free it. */
		free(sgi_buf);
		#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
			DEBUGMSG("Leave fill_sgi.");
		#endif
		return 0;
	}
	/* Allocate space for storing valid SGInfos. */
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tAllocating buffer space and copying SGInfo.");
	#endif
	*sgi = (SGInfo *)malloc(sizeof(SGInfo)*valid_sgi);
	memcpy(*sgi, sgi_buf, sizeof(SGInfo)*valid_sgi);
	/* Done with the temporary buffer, free it. */
	free(sgi_buf);
	/* Sort SGInfo array according to second / frames
	 */
	qsort((void *)*sgi, valid_sgi, sizeof(SGInfo), compare_sgi);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave fill_sgi.");
	#endif
	return valid_sgi;
}

/* 
 * Create an SGInfo instance for the given filename.
 * Arguments:
 *   void *arg -- Pointer to filename string.
 * Returns:
 *   void * -- Pointer to SGInfo instance if the filename produced a 
 *     valid sg_access result (test on SGInfo.smi) or NULL if not.
 * Notes:
 *   This method is suitable for a call via pthread_create.
 */
static void * sgthread_fill_sgi(void *arg)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter sgthread_fill_sgi.");
	#endif
	#ifdef DEBUG_LEVEL
		char _dbgmsg[0x200];
	#endif
	char *filename = (char *)arg; // filename to try to access
	SGInfo *sgi = (SGInfo *)calloc(sizeof(SGInfo), 1); // SGInfo pointer to return
	sgi->name = NULL;
	sgi->verbose = 0;
	sg_open(filename,sgi);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ snprintf(_dbgmsg,0x200,"\tsgi->smi.mmfd = %d",sgi->smi.mmfd);
		//~ DEBUGMSG(_dbgmsg);
		//~ sg_report(sgi,"\tSG Report:");
		DEBUGMSG("Leave sgthread_fill_sgi.");
	#endif
	return (void *)sgi;
}

// Reading VDIF from SG files
/*
 * Read one block's worth of VDIF frames from a group of SG files.
 * Arguments:
 *   SGInfo *sgi -- Array of valid SGInfo instances (i.e. had to have
 *     been accessed prior to calling this method).
 *   int n_sgi -- Number of SGInfo elements in the array.
 *   off_t iblock -- The block index.
 *   uint32_t **vdif_buf -- Address of a pointer which can be used to
 *     store the location of the VDIF buffer created and filled.
 * Returns:
 *   int -- The number of VDIF frames contained in the buffer.
 * Notes:
 *   This method creates as many threads as there are SGInfo elements in 
 *     the array, using pthread_create with sgthread_read_block as the
 *     thread start method.
 *   The VDIF buffer size is determined by counting the packets per 
 *     block total for all SGInfo instances, although the actual used
 *     size may be smaller if one of the blocks is short.
 */
int read_block_vdif_frames(SGInfo *sgi, int n_sgi, off_t iblock, uint32_t **vdif_buf)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter read_block_vdif_frames.");
	#endif
	/* Messages to SG threads. Need one for each created thread, since
	 * multiple threads may be using their messages simultaneously.
	 */
	MsgToSGThread msg_out[n_sgi]; // messages to sgthread
	/* Store messages received from SG threads. Need only one, since it
	 * is filled with each call to pthread_join, which happens 
	 * sequentially in the main thread.
	 */
	MsgFromSGThread *msg_in; // message from sgthread
	int ithread; // thread counter
	int thread_result; // result of calls to pthread methods
	pthread_t sg_threads[n_sgi]; // the pthreads used
	
	int frames_estimate = 0; // estimate the size of buffer to create
	int frames_read = 0; // count the number of frames received
	int frame_size = sgi[0].pkt_size; // size of a frame
	
	/* Launch threads to read data */
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tLaunching threads.");
	#endif
	for (ithread=0; ithread<n_sgi; ithread++)
	{
		msg_out[ithread].sgi_ptr = &(sgi[ithread]);
		msg_out[ithread].iblock = iblock;
		thread_result = pthread_create(&(sg_threads[ithread]),NULL,&sgthread_read_block,&(msg_out[ithread]));
		if (thread_result != 0)
		{
			perror("Unable to create thread.");
			exit(EXIT_FAILURE);
		}
		frames_estimate += sgi[ithread].sg_wr_pkts;
	}
	//printf("Expect %d frames.\n",frames_estimate);
	/* Create storage buffer. */
	*vdif_buf = (uint32_t *)malloc(frames_estimate*frame_size);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tJoining threads.");
	#endif
	/* Join the threads and copy data. */
	for (ithread=0; ithread<n_sgi; ithread++)
	{
		thread_result = pthread_join(sg_threads[ithread],(void *)&msg_in);
		//printf("Thread %d read %d frames.\n",ithread,msg_in->num_frames);
		if (thread_result != 0)
		{
			perror("Unable to join thread.");
			exit(EXIT_FAILURE);
		}
		if (msg_in->num_frames > 0)
		{
			memcpy((void *)(*vdif_buf + frames_read*frame_size/sizeof(uint32_t)),(void *)(msg_in->data_buf),msg_in->num_frames*frame_size);
			frames_read += msg_in->num_frames;
		}
		else
		{
			//printf("Empty frame for thread %d.\n",ithread);
		}
		#ifdef DEBUG_NO_SG
			int jj;
			printf("%s:%d:DEBUG_NO_SG:msg_in->data_buf (ithread:%2d) = [",__FILE__,__LINE__,ithread);
			for (jj=0; jj<msg_in->num_frames; jj++)
			{
				if (jj > 0)
					printf(", ");
				printf("%5d",msg_in->data_buf[jj]);
				if (jj == 5)
				{
					jj = msg_in->num_frames-5;
					printf(", ... ");
				}
			}
			printf("]\n");
		#else
			
		#endif
		/* NOTE: Free the received message memory */
		free_msg_from_thread(msg_in);
	}
	//printf("Got %d frames.\n",frames_read);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave read_block_vdif_frames.");
	#endif
	/* Return the frame count. */
	return frames_read;
}

/*
 * Read one block's worth of VDIF packets from the given SG file.
 * Arguments:
 *   void *arg -- MsgToSGThread by reference that contains a pointer to 
 *     a valid SGInfo instance, and an index specifying which block to
 *     read.
 * Returns:
 *   void *msg_out -- MsgFromSGThread pointer which contains a data 
 *     buffer filled with VDIF data read from block, and a frame count.
 * Notes:
 *   This method is suitable for a call via pthread_create.
 */
static void * sgthread_read_block(void *arg)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter sgthread_read_block.");
	#endif
	MsgToSGThread *msg_in = (MsgToSGThread *)arg;
	MsgFromSGThread *msg_out = (MsgFromSGThread *)malloc(sizeof(MsgFromSGThread));
	uint32_t *start = NULL;
	uint32_t *end = NULL;
	int num_frames;
	int ii;
	uint32_t *data_buf; // pointer to VDIF data buffer
	#ifdef DEBUG_LEVEL
		char _dbgmsg[0x200];
	#endif
	
	// check if this is a valid block number
	if (msg_in->iblock >= msg_in->sgi_ptr->sg_total_blks) 
	{
		return (void *)(msg_out);
	}
	// reopen file
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		snprintf(_dbgmsg,0x200,"\tReopen file %s",msg_in->sgi_ptr->name);
		DEBUGMSG(_dbgmsg);
	#endif
	SGMMInfo *sgmmi = sg_reopen(msg_in->sgi_ptr);
	// get packet for the desired block
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		snprintf(_dbgmsg,0x200,"\tRead block %d",(int)msg_in->iblock);
		DEBUGMSG(_dbgmsg);
	#endif
	start = sg_pkt_by_blk(msg_in->sgi_ptr,msg_in->iblock,&num_frames,&end);
	msg_out->num_frames = num_frames;
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		snprintf(_dbgmsg,0x200,"\tFrames read is %d",num_frames);
		DEBUGMSG(_dbgmsg);
	#endif
	// allocate data storage and copy data to memory
	msg_out->data_buf = (uint32_t *)malloc(msg_out->num_frames*msg_in->sgi_ptr->pkt_size);
	if (msg_out->data_buf != NULL)
	{
		memcpy(msg_out->data_buf,start,0*sizeof(uint32_t));
	}
	// and close file
	sg_close(msg_in->sgi_ptr);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave sgthread_read_block.");
	#endif
	// return pointer to buffer
	return (void *)msg_out;
}

/*
 * Free the resources allocated for the given MsgFromSGThread instance.
 * Arguments:
 *   MsgFromSGThread *msg -- Pointer to the MsgFromSGThread instance 
 *     that needs to be freed.
 */
void free_msg_from_thread(MsgFromSGThread *msg)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter free_msg_from_thread.");
	#endif
	if (msg != NULL)
	{
		if (msg->data_buf != NULL)
		{
			free(msg->data_buf);
		}
		free(msg);
	}
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave free_msg_from_thread.");
	#endif
}

/*
 * Free the resources allocated for an SGInfo structure.
 * Arguments:
 *   SGInfo *sgi -- Pointer to allocated SGInfo structure.
 */
void free_sg_info(SGInfo *sgi)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter free_sg_info.");
	#endif
	if (sgi != NULL)
	{
		if (sgi->name != NULL)
		{
			free(sgi->name);
		}
		free(sgi);
	}
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave free_sg_info.");
	#endif
}
