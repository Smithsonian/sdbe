#include <getopt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>

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

#define VDIF_FRAMES_PER_SECOND 125000

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

#define LAST_VDIF_SECS_INRE(a) ((VDIFHeader *)(&(a->data_buf[(a->n_frames-1)*(a->sgi->pkt_size)/sizeof(uint32_t)])))->w1.secs_inre
#define FIRST_VDIF_SECS_INRE(a) ((VDIFHeader *)(a->data_buf))->w1.secs_inre
#define LAST_VDIF_DF_NUM_INSEC(a) ((VDIFHeader *)(&(a->data_buf[(a->n_frames-1)*(a->sgi->pkt_size)/sizeof(uint32_t)])))->w2.df_num_insec
#define FIRST_VDIF_DF_NUM_INSEC(a) ((VDIFHeader *)(a->data_buf))->w2.df_num_insec

/* Structures for SG thread communications. */
typedef struct msg_to_sgthread { // message TO an SG thread
	SGInfo *sgi_ptr; // points to SGInfo 
	off_t iblock; // number block to read from
} MsgToSGThread;
typedef struct msg_from_sgthread { // message FROM an SG thread
	uint32_t *data_buf; // pointer to VDIF data buffer
	int num_frames; // number of frames in buffer
} MsgFromSGThread;

typedef struct sg_part {
	SGInfo *sgi; // points to SGInfo for single SG file
	off_t iblock; // next block to read from SG file
	uint32_t *data_buf; // points to start VDIF buffer from previous read
	uint32_t n_frames; // number of VDIF frames in buffer
} SGPart;

typedef struct sg_plan {
	int n_sgprt; // number of SGPart elements
	SGPart *sgprt; // array of SGPart elements (one per SG file)
} SGPlan;

#ifdef DEBUG_LEVEL
void print_sg_part(SGPart *sgprt, const char *label)
{
	printf("%sSGPart 0x%lx:",label,(unsigned long int)sgprt);
	if (sgprt->data_buf != NULL)
	{
		printf(" %u.%u -->> %u.%u",(uint32_t)FIRST_VDIF_SECS_INRE(sgprt),(uint32_t)FIRST_VDIF_DF_NUM_INSEC(sgprt),
			(uint32_t)LAST_VDIF_SECS_INRE(sgprt),(uint32_t)LAST_VDIF_DF_NUM_INSEC(sgprt));
		//~ printf("   (%u, %u)",*(sgprt->data_buf),*(sgprt->data_buf+1));
	}
	printf("\n");
	printf("%s\t.iblock = %lu\n",label,(unsigned long int)(sgprt->iblock));
	printf("%s\t.data_buf = 0x%lx\n",label,(unsigned long int)(sgprt->data_buf));
	printf("%s\t.n_frames = %d\n",label,sgprt->n_frames);
}

void print_sg_plan(SGPlan *sgpln, const char *label)
{
	int ii;
	char new_label[strlen(label)+2];
	snprintf(new_label, strlen(label)+2, "\t\t%s",label);
	printf("%sSGPlan 0x%lx:\n",label,(unsigned long int)sgpln);
	for (ii=0; ii<sgpln->n_sgprt; ii++)
	{
		print_sg_part(&(sgpln->sgprt[ii]),new_label);
	}
}
#endif

// Forward declarations
// Basic program utilities.
void initialize(void);
void parse_options(int argc, char **argv);
void print_usage(void);
void do_exit(void);
// SGInfo utilities
int mod_disk_id(int mod, int disk);
int compare_sg_info(const void *a, const void *b);
int compare_sg_part(const void *a, const void *b);
int make_sg_plan(SGPlan **sgpln, const char *pattern, int *mod_list, int n_mod, int *disk_list, int n_disk);
int map_sg_parts_contiguous(SGPlan *sgpln, int *mapping);
int test_sg_parts_contiguous(SGPart *a, SGPart *b);
// Reading VDIF from SG files
int read_block_vdif_frames(SGInfo *sgi, int n_sgi, off_t iblock, uint32_t **vdif_buf);
int read_next_block_vdif_frames(SGPlan *sgpln, uint32_t **vdif_buf);

// Thead methods
static void * sgthread_read_block(void *arg);
static void * sgthread_fill_sgi(void *arg);

// Memory management
void clear_sg_part_buffer(SGPart *sgprt);
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

	SGPlan *sgpln;
	uint32_t *vdif_buf = NULL;
	uint32_t vdif_buf_size = 0;
	
	int n_mod = 4;
	int mod_list[4] = { 1, 2, 3, 4 };
	int n_disk = 8;
	int disk_list[8] = { 1, 0, 2, 3, 4, 5, 6, 7 };
	
	/* Just some initialization.
	 */
	initialize();
	
	/* Parse input arguments and set global variables accordingly.
	 */
	parse_options(argc, argv);
	
	if (filename_stats != NULL)
	{
		fh_stats = fopen(filename_stats,"w");
		if (fh_stats == NULL)
		{
			printf("Unable to open diagnostics file.\n");
		}
		else
		{
			printf("Writing diagnostic data to %s.\n",filename_stats);
		}
	}

	
	/* Initialize the array of SGInfo structures by accessing each of 
	 * files that matches the specified input filename pattern. Also do
	 * other household tasks, like:
	 *   # Set the VDIF data payload size
	 */
	int ii, n_sgi;
	int progress = 0;
	printf("Filling SGInfos\n");
	n_sgi = make_sg_plan(&sgpln, file_pattern, mod_list, n_mod, disk_list, n_disk);
	printf("Found %d valid SG files.\n",n_sgi);
	//int n_sgi = read_block_vdif_frames(&sgi,&vdif_buf);
	if (n_sgi > 0)
	{
		uint32_t frame_size = sgpln->sgprt[0].sgi->pkt_size;
		uint32_t min_block = 0xFFFFFFFF;
		uint32_t max_block = 0x00000000;
		int sgi_count;
		for (sgi_count=0; sgi_count<n_sgi; sgi_count++)
		{
			if (sgpln->sgprt[sgi_count].sgi->sg_total_blks < min_block)
			{
				min_block = sgpln->sgprt[sgi_count].sgi->sg_total_blks;
			}
			if (sgpln->sgprt[sgi_count].sgi->sg_total_blks > max_block)
			{
				max_block = sgpln->sgprt[sgi_count].sgi->sg_total_blks;
			}
		}
		//~ min_block--; // in case short block
		printf("Found at least %u blocks per file.\n",min_block);
		printf("Total read size is %lu = %ub x %uP/b x %uB/P x %dd.\n",(unsigned long int)min_block*sgpln->sgprt[0].sgi->sg_wr_pkts*sgpln->sgprt[0].sgi->pkt_size*n_sgi, min_block, sgpln->sgprt[0].sgi->sg_wr_pkts, sgpln->sgprt[0].sgi->pkt_size, n_sgi);
		uint32_t total_frames = 0;
		uint32_t frame_incr = 0;
		uint32_t block_count;
		fprintf(stdout,"Progress: %3d%%",progress);
		fflush(stdout);
		for (block_count = 0; block_count<max_block; block_count++)
		{
			//~ total_frames += read_block_vdif_frames(sgi,n_sgi,block_count,&vdif_buf);
			frame_incr = read_next_block_vdif_frames(sgpln,&vdif_buf);
			if (frame_incr == 0)
			{
				break;
			}
			if (fh_stats != NULL)
			{
				for (ii=0; ii<frame_incr; ii++)
				{
					fwrite((void *)(vdif_buf+ii*frame_size/sizeof(uint32_t)),sizeof(uint32_t),2,fh_stats);
				}
			}
			total_frames += frame_incr;
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
				printf("Stats filename is %s\n",optarg);
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

int compare_int_descend(const void *a, const void *b)
{
	int *int_a = (int *)a;
	int *int_b = (int *)b;
	return *int_b < *int_a ? -1 : *int_b > *int_a;
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
int compare_sg_info(const void *a, const void *b)
{
	//~ #if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ DEBUGMSG("Enter compare_sg_info.");
	//~ #endif
	SGInfo *sgi_a = (SGInfo *)a;
	SGInfo *sgi_b = (SGInfo *)b;
	int result = sgi_a->first_secs < sgi_b->first_secs ? -1 : sgi_a->first_secs > sgi_b->first_secs;
	if (result == 0)
	{
		result = sgi_a->first_frame < sgi_b->first_frame ? -1 : sgi_a->first_frame > sgi_b->first_frame;
	}
	//~ #if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ DEBUGMSG("Leave compare_sg_info.");
	//~ #endif
	return result;
}

/*
 * Comparison method to sort an array of SGPart elements.
 * Arguments
 *   const void *a -- SGPart by reference.
 *   const void *b -- SGPart by reference.
 * Return:
 *   int -- Returns -1 if a < b, 0 if a == b, and 1 if a > b.
 * Notes:
 *   The comparison is based on the timestamp on the first VDIF frame in
 *     the a->data_buf and b->data_buf.
 */
int compare_sg_part(const void *a, const void *b)
{
	//~ #if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ DEBUGMSG("Enter compare_sg_part.");
		//~ char _dbgmsg[0x200];
	//~ #endif
	SGPart *sgprt_a = (SGPart *)a;
	SGPart *sgprt_b = (SGPart *)b;
	/* Seconds since reference epoch. */
	uint32_t secs_inre_a = FIRST_VDIF_SECS_INRE(sgprt_a);
	uint32_t secs_inre_b = FIRST_VDIF_SECS_INRE(sgprt_b);
	/* Data frame number within second */
	uint32_t df_num_insec_a = FIRST_VDIF_DF_NUM_INSEC(sgprt_a);
	uint32_t df_num_insec_b = FIRST_VDIF_DF_NUM_INSEC(sgprt_b);
	
	int result = secs_inre_a < secs_inre_b ? -1 : secs_inre_a > secs_inre_b;
	//~ printf("%d = %d ? %d : %d\n",result,secs_inre_a < secs_inre_b,-1,secs_inre_a > secs_inre_b);
	if (result == 0)
	{
		result = df_num_insec_a < df_num_insec_b ? -1 : df_num_insec_a > df_num_insec_b;
		//~ printf("%d = %d ? %d : %d\n",result,df_num_insec_a < df_num_insec_b,-1,df_num_insec_a > df_num_insec_b);
	}
	//~ #if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ print_sg_part(sgprt_a,"sgprt_a:\t");
		//~ print_sg_part(sgprt_b,"sgprt_b:\t");
		//~ snprintf(_dbgmsg,0x200,"Result = %d\n",result);
		//~ DEBUGMSG(_dbgmsg);
		//~ DEBUGMSG("Leave compare_sg_part.");
	//~ #endif
	return result;
}

/*
 * Find a contiguous mapping of SGParts in the given SGPlan.
 * Arguments:
 *   SGPlan *sgpln -- SGPlan that contains the SGParts array to order.
 *   int *mapping -- Allocated integer array that will contain the 
 *     ordered mapping so that the first M entries will list the 
 *     contiguous blocks from start to end. This is followed by 
 *     sgpln->n_sgprt-M negative indecies that list blocks that are not
 *     contiguous with this block set.
 * Returns:
 *   int -- The number of contiguous blocks found.
 */
int map_sg_parts_contiguous(SGPlan *sgpln, int *mapping)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter map_sg_parts_contiguous.");
	#endif
	int ii, jj;
	int idx_min_new = 0;
	int tmp_map = 0;
	int return_value = 0;
	int dead_nodes = sgpln->n_sgprt;
	/* Initialize mapping array. Just put indecies in increasing 
	 * magnitude, and make entries for dead nodes negative.
	 */
	for (ii=0; ii<sgpln->n_sgprt; ii++)
	{
		if (sgpln->sgprt[ii].n_frames > 0)
		{
			dead_nodes--;
			mapping[ii] = ii+1;
		}
		else
		{
			mapping[ii] = -(ii+1);
		}
	}
	/* If all nodes dead, just return zero. */
	if (dead_nodes == sgpln->n_sgprt)
	{
		return 0;
	}
	/* Put all dead nodes at the end */
	qsort((void *)mapping, sgpln->n_sgprt, sizeof(int), compare_int_descend);
	/* Sort according to timestamps */
	for (ii=0; ii<sgpln->n_sgprt-dead_nodes; ii++)
	{
		idx_min_new = ii;
		for (jj=ii; jj<sgpln->n_sgprt-dead_nodes; jj++)
		{
			if (compare_sg_part((void *)&(sgpln->sgprt[mapping[jj]-1]),(void *)&(sgpln->sgprt[mapping[idx_min_new]-1])) < 0)
			{
				idx_min_new = jj;
			}
		}
		tmp_map = mapping[ii];
		mapping[ii] = mapping[idx_min_new];
		mapping[idx_min_new] = tmp_map;
	}
	/* Check data continuity, and set index negative if not. */
	for (ii = 0;ii<sgpln->n_sgprt-dead_nodes-1; ii++)
	{
		if (!test_sg_parts_contiguous(&(sgpln->sgprt[mapping[ii]-1]),&(sgpln->sgprt[mapping[ii+1]-1])))
		{
			break;
		}
	}
	return_value = ii+1;
	for (jj=return_value; jj<sgpln->n_sgprt-dead_nodes; jj++)
	{
		mapping[jj] = -mapping[jj];
	}
	//~ printf("Mapping = [");
	//~ for (ii=0; ii<sgpln->n_sgprt; ii++)
	//~ {
		//~ printf("%5d",mapping[ii]);
	//~ }
	//~ printf("]\n");
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Leave map_sg_parts_contiguous.");
	#endif
	return return_value;
}

/*
 * Test whether two SG parts are contiguous.
 * Arguments:
 *   SGPart *a -- Pointer to SGPart assumed to contain first data.
 *   SGPart *b -- Pointer to SGPart assumed to contain last data.
 * Returns:
 *   int -- 1 if contiguous, 0 if not.
 * Notes:
 *   Continuinity means that the last VDIF frame in the a->data_buf
 *     and the first VDIF frame in b->data_buf are adjacent in time, 
 *     according to seconds-since-reference-epoch and data-frame-within-
 *     second counters.
 */
int test_sg_parts_contiguous(SGPart *a, SGPart *b)
{
	/* Seconds since reference epoch. */
	uint32_t secs_inre_a = LAST_VDIF_SECS_INRE(a);
	uint32_t secs_inre_b = FIRST_VDIF_SECS_INRE(b);
	/* Data frame number within second */
	uint32_t df_num_insec_a = LAST_VDIF_DF_NUM_INSEC(a);
	uint32_t df_num_insec_b = FIRST_VDIF_DF_NUM_INSEC(b);
	if (secs_inre_b == secs_inre_a)
	{
		if (df_num_insec_b == df_num_insec_a+1)
		{
			return 1;
		}
	}
	else if (secs_inre_b == secs_inre_a+1)
	{
		if (df_num_insec_a == VDIF_FRAMES_PER_SECOND-1 && df_num_insec_b == 0)
		{
			return 1;
		}
	}
	return 0;
}

/*
 * Allocate memory and fill it with SGInfo instances.
 * Arguments:
 *   SGPlan **sgplan -- Address of SGPlan pointer to allocate memory.
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
 *   The SGInfo entries stored in sgplan are sorted in ascending order
 *     according to the timestamp on the first VDIF frame in each SG 
 *     file.
 *   For each valid SG file encountered an SGPart element is stored in
 *     SGPlan.
 */
int make_sg_plan(SGPlan **sgpln, const char *pattern, int *mod_list, int n_mod, int *disk_list, int n_disk)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter make_sg_plan.");
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
				//~ sg_close(sgi_tmp);
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
			DEBUGMSG("Leave make_sg_plan.");
		#endif
		return 0;
	}
	/* Allocate space for storing valid SGInfos. */
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tAllocating buffer space and copying SGInfo.");
	#endif
	
	/* Sort SGInfo array according to second / frames */
	qsort((void *)sgi_buf, valid_sgi, sizeof(SGInfo), compare_sg_info);
	/* Allocate memory for SGPlan */
	*sgpln = (SGPlan *)malloc(sizeof(SGPlan));
	(*sgpln)->sgprt = (SGPart *)malloc(sizeof(SGPart)*valid_sgi);
	for (itmp=0; itmp<valid_sgi; itmp++)
	{
		// Allocate memory for SGInfo and copy
		(*sgpln)->sgprt[itmp].sgi = (SGInfo *)malloc(sizeof(SGInfo));
		memcpy((*sgpln)->sgprt[itmp].sgi, &(sgi_buf[itmp]), sizeof(SGInfo));
		// Initialize block counter, VDIF buffer, and frame counter
		(*sgpln)->sgprt[itmp].iblock = 0;
		(*sgpln)->sgprt[itmp].data_buf = NULL;
		(*sgpln)->sgprt[itmp].n_frames = 0;
	}
	(*sgpln)->n_sgprt = valid_sgi;
	/* Done with the temporary buffer, free it. */
	free(sgi_buf);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		print_sg_plan(*sgpln,"\t");
		DEBUGMSG("Leave make_sg_plan.");
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
	//~ #if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ DEBUGMSG("Enter sgthread_fill_sgi.");
		//~ char _dbgmsg[0x200];
	//~ #endif
	char *filename = (char *)arg; // filename to try to access
	SGInfo *sgi = (SGInfo *)calloc(sizeof(SGInfo), 1); // SGInfo pointer to return
	sgi->name = NULL;
	sgi->verbose = 0;
	sg_open(filename,sgi);
	//~ #if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ snprintf(_dbgmsg,0x200,"\tsgi->smi.mmfd = %d",sgi->smi.mmfd);
		//~ DEBUGMSG(_dbgmsg);
		//~ sg_report(sgi,"\tSG Report:");
		//~ DEBUGMSG("Leave sgthread_fill_sgi.");
	//~ #endif
	return (void *)sgi;
}

/*
 * Read the next block of VDIF frames.
 * Arguments:
 *   SGPlan *sgpln -- The SGPlan created for a given filename pattern.
 *   uint32_t **data_buf -- Address of pointer which can be used to 
 *     store the location of the data buffer created and filled by 
 *     reading the next block.
 * Returns:
 *   int -- The number of VDIF frames contained in the buffer, zero if
 *     end of all files reached, and -1 if the data is no longer 
 *     contiguous.
 * Notes:
 *   This method attempts to read a contiguous set of VDIF frames that
 *     is the equivalent of one SG block per SG file contained in the SG
 *     plan. Blocks from different files are stitched together such that
 *     the first frame in one block directly follows the last frame of
 *     another block. Blocks with data that do not flow contiguously 
 *     from the first frame for the current block are stored in the 
 *     buffer of the associated SGPart, inside SGPlan. Upon subsequent 
 *     calls to this method no further blocks of data is read from that 
 *     particular SG file until its block can be stitched togther with 
 *     the contiguous flow.
 *   Block counter for each SGPart is updated if frames where read from
 *     that file.
 */
int read_next_block_vdif_frames(SGPlan *sgpln, uint32_t **vdif_buf)
{
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Enter read_next_block_vdif_frames.");
		print_sg_plan(sgpln,"\t");
		char _dbgmsg[0x200];
	#endif
	int ithread; // thread counter
	int thread_result; // result of calls to pthread methods
	pthread_t sg_threads[sgpln->n_sgprt]; // the pthreads used
	int sg_threads_mask[sgpln->n_sgprt];
	
	int frames_estimate = 0; // estimate the size of buffer to create
	int frames_read = 0; // count the number of frames received
	int frame_size = sgpln->sgprt[0].sgi->pkt_size; // size of a frame
	
	int isgprt;
	int n_contiguous_blocks = 0;
	int mapping[sgpln->n_sgprt];
	
	/* Launch threads to read data */
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tLaunching threads.");
	#endif
	for (ithread=0; ithread<sgpln->n_sgprt; ithread++)
	{
		/* For each SGPart, check if its data buffer is empty, which 
		 * indicates that the next block of data should be read.
		 */
		sg_threads_mask[ithread] = 0;
		if (sgpln->sgprt[ithread].n_frames == 0 && sgpln->sgprt[ithread].iblock < sgpln->sgprt[ithread].sgi->sg_total_blks)
		{
			sg_threads_mask[ithread] = 1;
			thread_result = pthread_create(&(sg_threads[ithread]),NULL,&sgthread_read_block,&(sgpln->sgprt[ithread]));
			if (thread_result != 0)
			{
				perror("Unable to create thread.");
				exit(EXIT_FAILURE);
			}
			frames_estimate += sgpln->sgprt[ithread].sgi->sg_wr_pkts;
		}
	}
	/* Create storage buffer. Assume that the number of frames read
	 * is always smaller than or equal to the number of estimated frames
	 */
	*vdif_buf = (uint32_t *)malloc(frames_estimate*frame_size);
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("\tJoining threads.");
	#endif
	/* Join the threads */
	for (ithread=0; ithread<sgpln->n_sgprt; ithread++)
	{
		/* Only join threads that have been started. */
		if (sg_threads_mask[ithread] == 1)
		{
			thread_result = pthread_join(sg_threads[ithread],NULL);
			//printf("Thread %d read %d frames.\n",ithread,msg_in->num_frames);
			if (thread_result != 0)
			{
				perror("Unable to join thread.");
				exit(EXIT_FAILURE);
			}
			/* If we read frames from this SG file, update the block 
			 * counter.
			 */
			if (sgpln->sgprt[ithread].n_frames > 0)
			{
				sgpln->sgprt[ithread].iblock++;
			}
		}
	}
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		print_sg_plan(sgpln,"\t");
	#endif
	
/***********************************************************************
 * For now, ignore discontinuity in the data.
 *
 */
	n_contiguous_blocks = map_sg_parts_contiguous(sgpln, mapping);
	if (n_contiguous_blocks == 0)
	{
		printf("No contiguous blocks found.\n");
		return 0;
	}
	for (isgprt=0; isgprt<n_contiguous_blocks; isgprt++)
	{
		memcpy((void *)(*vdif_buf + frames_read*frame_size/sizeof(uint32_t)),
				(void *)(sgpln->sgprt[mapping[isgprt]-1].data_buf),sgpln->sgprt[mapping[isgprt]-1].n_frames*frame_size);
		frames_read += sgpln->sgprt[mapping[isgprt]-1].n_frames;
		clear_sg_part_buffer(&(sgpln->sgprt[mapping[isgprt]-1]));
	}
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		snprintf(_dbgmsg,0x200,"Found %d contiguous blocks\n",n_contiguous_blocks);
		DEBUGMSG(_dbgmsg);
	#endif
 /*
 *
 **********************************************************************/
	//~ for (ithread=0; ithread<sgpln->n_sgprt; ithread++)
	//~ {
		//~ if (sg_threads_mask[ithread] == 1 && sgpln->sgprt[ithread].n_frames > 0)
		//~ {
			//~ memcpy((void *)(*vdif_buf + frames_read*frame_size/sizeof(uint32_t)),
					//~ (void *)(sgpln->sgprt[ithread].data_buf),sgpln->sgprt[ithread].n_frames*frame_size);
			//~ frames_read += sgpln->sgprt[ithread].n_frames;
			//~ clear_sg_part_buffer(&(sgpln->sgprt[ithread]));
		//~ }
	//~ } 
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		print_sg_plan(sgpln,"\t");
		DEBUGMSG("Leave read_block_vdif_frames.");
	#endif
	/* Return the frame count. */
	return frames_read;
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
	//~ #if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		//~ DEBUGMSG("Enter sgthread_read_block.");
	//~ #endif
	SGPart *sgprt = (SGPart *)arg;
	uint32_t *start = NULL;
	uint32_t *end = NULL;
	int ii;
	uint32_t *data_buf; // pointer to VDIF data buffer
	uint32_t n_frames;
	//~ #ifdef DEBUG_LEVEL
		//~ char _dbgmsg[0x200];
	//~ #endif
	
	// check if this is a valid block number
	if (sgprt->iblock < sgprt->sgi->sg_total_blks) 
	{
		//~ start = sg_pkt_by_blk(sgprt->sgi,0,&(sgprt->n_frames),&end);
		start = sg_pkt_by_blk(sgprt->sgi,sgprt->iblock,&(sgprt->n_frames),&end);
		// allocate data storage and copy data to memory
		sgprt->data_buf = (uint32_t *)malloc(sgprt->n_frames*sgprt->sgi->pkt_size);
		if (sgprt->data_buf != NULL)
		{
			memcpy(sgprt->data_buf,start,sgprt->n_frames*sgprt->sgi->pkt_size);
		}
	}
	return NULL;
}

/*
 * Clear the data buffer in SGPart.
 * Arguments:
 *   SGPart *sgprt -- Pointer to SGPart instance.
 * Return:
 *   void
 */
void clear_sg_part_buffer(SGPart *sgprt)
{
	sgprt->n_frames = 0;
	if (sgprt->data_buf != NULL)
	{
		free(sgprt->data_buf);
		sgprt->data_buf = NULL;
	}
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
