#include <stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include "sg_access.h"
#include<time.h>

#define DEBUG_LEVEL_DEBUG 40
#define DEBUG_LEVEL_INFO 30
#define DEBUG_LEVEL_WARNING 20
#define DEBUG_LEVEL_ERROR 10
#define DEBUG_LEVEL DEBUG_LEVEL_WARNING

#define N_MOD 4
#define N_DISK 8

#ifdef DEBUG_LEVEL
	void debug_msg(const char *msg, const char *filename, int linenum);
	#define DEBUGMSG(m) debug_msg(m,__FILE__,__LINE__)

	void error_msg(const char *msg, const char *filename, int linenum);
	#define ERRORMSG(m) error_msg(m,__FILE__,__LINE__)

	void info_msg(const char *msg, const char *filename, int linenum);
	#define INFOMSG(m) info_msg(m,__FILE__,__LINE__)
#endif

void print_usage(void);

int mod_disk_id(int mod, int disk);

int main(int argc, char **argv)
{
	SGInfo sgi[N_MOD*N_DISK];
	char filename[0x100];
	struct timespec t0_inner,t1_inner,t0_outer,t1_outer;
	uint32_t nblocks = 1;
	FILE *fh_output = NULL;
	
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Start.");
	#endif
	if (argc < 2)
	{
		#if DEBUG_LEVEL >= DEBUG_LEVEL_ERROR
			ERRORMSG("No input file specified.");
		#endif
		print_usage();
	}
	else
	{
		if (argc > 2)
		{
			nblocks = atoi(argv[2]);
			if (argc > 3)
			{
				fh_output = fopen(argv[3],"w");
			}
		}
		clock_gettime(CLOCK_REALTIME,&t0_outer);
		uint32_t imod,idisk,iblock;
		uint32_t pkt_count = 0;
		uint32_t block_counts[32] = { 1 };
		
		
		float completion = 0;
		for (iblock=0; iblock<nblocks; iblock++)
		{
			for (imod=1; imod<5; imod++)
			{
				for (idisk=0; idisk<8; idisk++)
				{
					int id = mod_disk_id(imod,idisk);
					if (iblock == 0)
					{
						snprintf(filename,0x100,"/mnt/disks/%u/%u/data/%s",imod,idisk,argv[1]);
						#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
							char msg[0x200];
							snprintf(msg,0x200,"Reading from file '%s'.",filename);
							INFOMSG(msg);
						#endif
						sgi[id].name = NULL;
						sgi[id].verbose = 0;
						sg_open(filename,&(sgi[id]));
						block_counts[id] = sgi[id].sg_total_blks;
						sg_report(&(sgi[id]),"\tSG report:");
						if (id == N_MOD*N_DISK-1)
						{
							fprintf(stdout,"\nCompleted: %3d%%",(int)completion);
							fflush(stdout);
						}
					}

					// read packets in a block
					if (iblock < block_counts[id])
					{
						uint32_t *end;
						int nl;
						uint32_t *start = sg_pkt_by_blk(&(sgi[id]),0,&nl,&end);
						//printf("First block packets in %u --> %u\n",start,end);
						pkt_count += (end-start)/2056;
						//uint32_t *vdif_buf = malloc((end-start)*sizeof(uint32_t));
						uint32_t idx_out=0,idx_in;
						//clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t0_inner);
						//for (idx_in=0; idx_in<(uint32_t)(end-start); idx_in++)
						//{
						//	vdif_buf[idx_out] = start[idx_in];
						//}
						if (fh_output != NULL)
						{
							if (iblock == 0 || iblock == nblocks-1)
							{
								fwrite((void *)&(start[idx_in]),sizeof(uint32_t),(end-start),fh_output);
							}
						}
						//free(vdif_buf);
						//clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t1_inner);
						//printf("\tReading done in %10.6fms\n",1e3*(double)(t1_inner.tv_sec - t0_inner.tv_sec) + 1e-6*(double)(t1_inner.tv_nsec - t0_inner.tv_nsec));
					}
					if (id == N_MOD*N_DISK-1)
					{
						
						completion = 100*(iblock+1)/nblocks;
						fprintf(stdout,"%c%c%c%c%3d%%",(char)8,(char)8,(char)8,(char)8,(int)completion);
						fflush(stdout);
						if (iblock == nblocks-1)
						{
							fprintf(stdout,"\n\n");
							fflush(stdout);
						}
					}
					if (iblock == nblocks-1)
					{
						#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
							char msg[0x200];
							snprintf(msg,0x200,"Closing file '%s'.",sgi[id].name);
							INFOMSG(msg);
						#endif
						sg_close(&(sgi[id]));
					}
				}
			}
		}
		clock_gettime(CLOCK_REALTIME,&t1_outer);
		printf("Total reading done in %10.6fms (%u packets)\n",1e3*(double)(t1_outer.tv_sec - t0_outer.tv_sec) + 1e-6*(double)(t1_outer.tv_nsec - t0_outer.tv_nsec),pkt_count);
		if (fh_output != NULL)
		{
			fclose(fh_output);
		}
	}
	#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
		DEBUGMSG("Stop.");
	#endif
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

	void info_msg(const char *msg, const char *filename, int linenum)
	{
	        printf("%s:%d:INFO:%s\n",filename,linenum,msg);
	}
#endif

void print_usage()
{
	printf("Usage: sgtool FILEPATTERN [NBLOCKS [OUTPUT]]\n");
	printf("\n");
	printf("\tReads NBLOCKS from each file named as /mnt/disks/<mod>/<disk>/data/FILEPATTERN where:\n");
	printf("\t\tmod = 1|2|3|4\n");
	printf("\t\tdisks = 0|1|2|3|4|5|6|7.\n");
	printf("\tIf specified, output is written to the file OUTPUT.\n");
	printf("\n\n");
}

int mod_disk_id(int mod, int disk)
{
	int id = (mod-1)*N_DISK + disk;
	return id;
}
