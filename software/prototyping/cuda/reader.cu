/*
 * GPU kernel code for the following pipeline components:
 *   VDIF interpreter
 *   B-engine depacketizer
 *   Pre-preprocessor
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cufft.h>

// VDIF constants
#define VDIF_BYTE_SIZE 1056 // VDIF frame size in bytes
#define VDIF_BYTE_SIZE_HEADER 32 // VDIF header size in bytes
#define VDIF_BYTE_SIZE_DATA 1024 // VDIF data size in bytes
#define VDIF_INT_SIZE (1056/4) // VDIF frame size in int
#define VDIF_INT_SIZE_HEADER (32/4) // VDIF header size in int
#define VDIF_INT_SIZE_DATA (1024/4) // VDIF data size in int
#define VDIF_BIT_DEPTH 2 // bits-per-sample

// GPU Controls
#define THREADS_PER_BLOCK_X 8 // set specifically so that each thread reads one int32_t in header
#define THREADS_PER_BLOCK_Y 4 // set so there is one warp per block, not necessarily optimal
//~ #define BLOCKS_PER_GRID 128 // arbitrary power of 2

// Data formats
#define BENG_CHANNELS_ 16384
#define BENG_CHANNELS (BENG_CHANNELS_+1) // number of channels PLUS added sample-rate/2 component for the complex-to-real inverse transform
#define BENG_SNAPSHOTS 128
#define BENG_BUFFER_IN_COUNTS 32 // we will buffer 32 B-engine frames
#define BENG_BUFFER_INDEX_MASK (BENG_BUFFER_IN_COUNTS-1) // mask used to convert B-engine counter to index into buffer
#define SWARM_N_FIDS 8
#define SWARM_XENG_PARALLEL_CHAN 8

// VDIF packed B-engine packet
#define BENG_VDIF_HDR_0_OFFSET_INT 4 // b1 b2 b3 b4
#define BENG_VDIF_HDR_1_OFFSET_INT 5 //  c  0  f b0
#define BENG_VDIF_CHANNELS_PER_INT 4 // a-d in a single uint32_t, and e-h in a single uint32_t
#define BENG_VDIF_INT_PER_SNAPSHOT (SWARM_XENG_PARALLEL_CHAN/BENG_VDIF_CHANNELS_PER_INT)
#define BENG_PACKETS_PER_FRAME (BENG_CHANNELS_/SWARM_XENG_PARALLEL_CHAN)
#define BENG_FRAME_COMPLETION_COMPLETE (BENG_PACKETS_PER_FRAME*THREADS_PER_BLOCK_X) // value of completion counter when B-engine frame complete, multiplication by THREADS_PER_BLOCK_x required since all x-threads increment counter

// Debugging
//~ #define DEBUG
//~ #define DEBUG_GPU
//~ #define DEBUG_GPU_CONDITION (blockIdx.x == 0 && threadIdx.x == 7 && threadIdx.y == 3)

/*
 *  Forward declarations
 */
// GPU kernels
__global__ void vdif_reader(
	uint32_t *vdif_frames, 
	uint32_t *fid_out, 
	uint32_t *cid_out, 
	uint32_t *bcount_out, 
	cufftComplex *beng_data_out_0,
	cufftComplex *beng_data_out_1, 
	uint32_t *beng_frame_completion,
	int32_t num_vdif_frames, 
	int blocks_per_grid);
// host utilities
void error_check(void);
void error_check(cudaError_t err);

int main(int argc, char **argv)
{
	#ifdef DEBUG
	printf("reader:DEBUG:Start\n");
	#endif
	
	// misc
	bool verbose_output = 0, logging = 0;
	int blocks_per_grid = 128;
	char filename_log[0x100] = "\0";
	FILE *fh_log = NULL;
	int repeats = 1, ir;
	int ii,ij,ik,il;
	time_t wall_clock;
	
	// input (host)
	FILE *fh = NULL;
	char filename_input[0x100] = "\0";
	int32_t num_vdif_frames = 0;
	uint32_t *vdif_buf = NULL;
	
	// input (device)
	uint32_t *gpu_vdif_buf = NULL;
	
	// output (host)
	uint32_t *fid;
	uint32_t *cid;
	uint32_t *bcount;
	cufftComplex *beng_data_0,*beng_data_1;
	
	// output (device)
	uint32_t *gpu_fid;
	uint32_t *gpu_cid;
	uint32_t *gpu_bcount;
	cufftComplex *gpu_beng_data_0,*gpu_beng_data_1;
	
	// control (host)
	uint32_t *beng_frame_completion; // counts number of packets received per b-count value
	
	// control (device)
	uint32_t *gpu_beng_frame_completion; // counts number of packets received per b-count value
	
	// for CUDA error checking
	cudaError_t err;
	
	// for CUDA timing
	struct timespec t0,t1;
	cudaEvent_t start, stop;
	float time_spent;
	
	#ifdef DEBUG
	printf("reader:DEBUG:Parse input\n");
	#endif
	
	int c;
	while (1)
	{
		int option_index = 0;
		static struct option long_options[] =
		{
			{  "input", required_argument, 0, 0 },
			{  "count", required_argument, 0, 0 },
			{"verbose",       no_argument, 0, 0 },
			{ "blocks", required_argument, 0, 0 },
			{"logfile", required_argument, 0, 0 },
			{"repeats", required_argument, 0, 0 },
			{        0,                 0, 0, 0 }
		};
		
		c = getopt_long(argc, argv, "i:c:vb:l:r:", long_options, &option_index);
		
		if (c == -1)
		{
			break;
		}
		
		if (c == 0)
		{
			switch (option_index)
			{
				case 0:
					c = 'i';
					break;
				case 1:
					c = 'c';
					break;
				case 2:
					c = 'v';
					break;
				case 3:
					c = 'b';
					break;
				case 4:
					c = 'l';
					break;
				case 5:
					c = 'r';
					break;
			}
		}
		
		switch (c)
		{
			case 'i':
				#ifdef DEBUG
				printf("\tInput file is", long_options[option_index].name);
				if (optarg)
				{
					printf(" '%s'.", optarg);
				}
				printf("\n");
				#endif
				snprintf(filename_input, sizeof(filename_input), "%s", optarg);
				break;
			case 'c':
				#ifdef DEBUG
				printf("\tReading", long_options[option_index].name);
				if (optarg)
				{
					printf(" %d", atoi(optarg));
				}
				printf(" VDIF frames.\n");
				#endif
				num_vdif_frames = atoi(optarg);
				break;
			case 'v':
				#ifdef DEBUG
				printf("\tVerbose output.\n");
				#endif
				verbose_output = 1;
				break;
			case 'b':
				#ifdef DEBUG
				printf("\tUsing ");
				if (optarg)
				{
					printf(" %d", atoi(optarg));
				}
				printf(" blocks per grid.\n");
				#endif
				blocks_per_grid = atoi(optarg);
				break;
			case 'l':
				#ifdef DEBUG
				printf("\tLogfile is", long_options[option_index].name);
				if (optarg)
				{
					printf(" '%s'.", optarg);
				}
				printf("\n");
				#endif
				snprintf(filename_log, sizeof(filename_log), "%s", optarg);
				logging = 1;
				break;
			case 'r':
				#ifdef DEBUG
				printf("\tRunning ");
				if (optarg)
				{
					printf(" %d", atoi(optarg));
				}
				printf(" repeats.\n");
				#endif
				repeats = atoi(optarg);
				break;
			default:
				fprintf(stderr,"?? getopt returned character code 0%o ??\n.",c);
				exit(EXIT_FAILURE);
		}
	}
	
	#ifdef DEBUG
	printf("reader:DEBUG:Reading %d VDIF frames from '%s'.\n",num_vdif_frames,filename_input);
	#endif
	
	// open logfile
	if (logging)
	{
		if (strlen(filename_input) == 0)
		{
			fprintf(stderr,"Log filename not specified.\n");
			exit(EXIT_FAILURE);
		}
		else
		{
			fh_log = fopen(filename_log,"a");
			if (fh_log != NULL)
			{
				if (ftell(fh_log) == 0)
				{
					wall_clock = time(NULL);
					fprintf(fh_log,"#log-file created: %s",ctime(&wall_clock)); // ctime returns string with \n included
					fprintf(fh_log,"#    CPU [ms]    CUDA [ms]\n");
				}
				fprintf(fh_log,"#Repeats: %d\n#Blocks per grid: %d\n",repeats,blocks_per_grid);
			}
			else
			{
				fprintf(stderr,"Unable to open logfile '%s'.\n",filename_log);
				exit(EXIT_FAILURE);
			}
		}
	}
	
	// read requested number of VDIF frames from input file
	if (num_vdif_frames < 1)
	{
		fprintf(stderr,"Number of frames has to be greater than 0 (given %d).\n",num_vdif_frames);
		exit(EXIT_FAILURE);
	}
	size_t num_vdif_bytes = num_vdif_frames*VDIF_BYTE_SIZE; // total bytes to read
	if (strlen(filename_input) == 0)
	{
		fprintf(stderr,"Input filename not specified.\n");
		exit(EXIT_FAILURE);
	}
	else
	{
		fh = fopen(filename_input,"r");
		if (fh != NULL)
		{
			// read file
			vdif_buf = (uint32_t *)malloc(num_vdif_bytes);
			if (vdif_buf == NULL)
			{
				fprintf(stderr,"Unable to allocate memory for input data.\n");
				fclose(fh);
				exit(EXIT_FAILURE);
			}
			#ifdef DEBUG
			printf("reader:DEBUG:Reading %d x %d = %d bytes from file.\n",num_vdif_frames,num_vdif_bytes/num_vdif_frames,num_vdif_bytes);
			#endif
			size_t num_elem = fread((void *)vdif_buf, VDIF_BYTE_SIZE, num_vdif_frames, fh); 
			if (num_elem != num_vdif_frames)
			{
				fprintf(stderr,"Unable to read all the requested data.\n");
				fclose(fh);
				exit(EXIT_FAILURE);
			}
			fclose(fh);
		}
		else
		{
			fprintf(stderr,"Unable to open input file '%s'.\n",filename_input);
			exit(EXIT_FAILURE);
		}
	}
	
	#ifdef DEBUG
	printf("reader:DEBUG:Creating CUDA events for timing.\n");
	#endif
	err = cudaEventCreate(&start);
	error_check(err);
	err = cudaEventCreate(&stop);
	error_check(err);
	
	#ifdef DEBUG
	printf("reader:DEBUG:Copying data from host to device...");
	#endif
	err = cudaMalloc((void **)&gpu_vdif_buf, num_vdif_bytes);
	error_check(err);
	err = cudaMemcpy(gpu_vdif_buf, vdif_buf, num_vdif_bytes, cudaMemcpyHostToDevice);
	error_check(err);
	#ifdef DEBUG
	printf(" done.\n");
	#endif
	
	// define control structures
	#ifdef DEBUG
	printf("reader:DEBUG:Allocating memory for control...");
	#endif
	beng_frame_completion = (uint32_t *)malloc(sizeof(uint32_t)*BENG_BUFFER_IN_COUNTS);
	err = cudaMalloc((void **)&gpu_beng_frame_completion, sizeof(uint32_t)*BENG_BUFFER_IN_COUNTS);
	error_check(err);
	#ifdef DEBUG
	printf(" done.\n");
	#endif
	// initialize completion counter on host and device
	for (ii=0; ii<BENG_BUFFER_IN_COUNTS; ii++)
	{
		beng_frame_completion[ii] = 0;
	}
	#ifdef DEBUG
	printf("reader:DEBUG:Copying control from host to device...");
	#endif
	err = cudaMemcpy(gpu_beng_frame_completion, beng_frame_completion, sizeof(uint32_t)*BENG_BUFFER_IN_COUNTS, cudaMemcpyHostToDevice);
	error_check(err);
	#ifdef DEBUG
	printf(" done.\n");
	#endif
	
	// output
	#ifdef DEBUG
	printf("reader:DEBUG:Allocating memory for output.\n");
	#endif
	cid = (uint32_t *)malloc(num_vdif_frames*sizeof(uint32_t));
	fid = (uint32_t *)malloc(num_vdif_frames*sizeof(uint32_t));
	bcount = (uint32_t *)malloc(num_vdif_frames*sizeof(uint32_t));
	
	size_t beng_data_bytes = BENG_CHANNELS*BENG_SNAPSHOTS*BENG_BUFFER_IN_COUNTS*sizeof(cuComplex);
	#ifdef DEBUG
	printf("reader:DEBUG:Allocating %d bytes for B-engine buffer.\n",beng_data_bytes);
	#endif
	beng_data_0 = (cufftComplex *)malloc(beng_data_bytes);
	beng_data_1 = (cufftComplex *)malloc(beng_data_bytes);
	err = cudaMalloc((void **)&gpu_cid, num_vdif_frames*sizeof(uint32_t));
	error_check(err);
	err = cudaMalloc((void **)&gpu_fid, num_vdif_frames*sizeof(uint32_t));
	error_check(err);
	err = cudaMalloc((void **)&gpu_bcount, num_vdif_frames*sizeof(uint32_t));
	error_check(err);
	err = cudaMalloc((void **)&gpu_beng_data_0, beng_data_bytes);
	error_check(err);
	err = cudaMalloc((void **)&gpu_beng_data_1, beng_data_bytes);
	error_check(err);
	
	#ifdef DEBUG
	printf("reader:DEBUG:Defining threads and blocks.\n");
	#endif
	dim3 threadsPerBlock(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y);
	#ifdef DEBUG
	printf("\tthreads-per-block = (%d,%d,%d)\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	#endif
	dim3 blocksPerGrid(blocks_per_grid);
	#ifdef DEBUG
	printf("\t  blocks-per-grid = (%d,%d,%d)\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);
	#endif
	
	
	for (ir=0; ir<repeats; ir++)
	{
		#ifdef DEBUG
		printf("reader:DEBUG:Call to GPU kernel.\n");
		#endif
		cudaEventRecord(start);
		cudaEventSynchronize(start);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t0);
		vdif_reader<<<blocksPerGrid,threadsPerBlock>>>(gpu_vdif_buf,gpu_fid,gpu_cid,gpu_bcount,gpu_beng_data_0,gpu_beng_data_1,gpu_beng_frame_completion,num_vdif_frames,blocks_per_grid);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t1);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_spent, start, stop);
		if (logging)
		{
			fprintf(fh_log,"   %10.6f",time_spent);
			fprintf(fh_log,"   %10.6f\n",1e6*(double)(t1.tv_sec - t0.tv_sec) + 1e-6*(double)(t1.tv_nsec - t0.tv_nsec));
		}
		else
		{
			printf("Reading VDIF frames finished in:\n\tCUDA: %10.6fms\n",time_spent);
			printf("\t CPU: %10.6fms\n",1e6*(double)(t1.tv_sec - t0.tv_sec) + 1e-6*(double)(t1.tv_nsec - t0.tv_nsec));
		}
	}
	
	#ifdef DEBUG
	printf("reader:DEBUG:Copying data from device to host...");
	#endif
	err = cudaMemcpy(fid, gpu_fid, num_vdif_frames*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	error_check(err);
	err = cudaMemcpy(cid, gpu_cid, num_vdif_frames*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	error_check(err);
	err = cudaMemcpy(bcount, gpu_bcount, num_vdif_frames*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	error_check(err);
	err = cudaMemcpy(beng_data_0, gpu_beng_data_0, num_vdif_frames*(VDIF_BYTE_SIZE_DATA*8/VDIF_BIT_DEPTH)*sizeof(float), cudaMemcpyDeviceToHost);
	error_check(err);
	err = cudaMemcpy(beng_data_1, gpu_beng_data_1, num_vdif_frames*(VDIF_BYTE_SIZE_DATA*8/VDIF_BIT_DEPTH)*sizeof(float), cudaMemcpyDeviceToHost);
	error_check(err);
	#ifdef DEBUG
	printf(" done.\n");
	#endif
	
	if (verbose_output)
	{
		cufftComplex beng_sample;
		for (ii=0; ii<num_vdif_frames; ii++)
		{
			fprintf(stdout,"B-engine packet: fid=%d, cid=%d, bcount=%d\n",fid[ii],cid[ii],bcount[ii]);
		}
		for (il=0; il<2; il++) // loop over 2 phased sums
		{
			fprintf(stdout,"Phased sum %d:\n",il);
			for (ii=0; ii<BENG_BUFFER_IN_COUNTS; ii++) // for each B-engine frame
			{
				fprintf(stdout,"\tB-eng buffer %d:\n",ii);
				for (ij=0; ij<BENG_SNAPSHOTS; ij++) // for each snapshot
				{
					fprintf(stdout,"\t\tB-eng snapshot %d:\n",ij);
					fprintf(stdout,"\t\t\tData:\n");
					for (ik=0; ik<BENG_CHANNELS; ik++) // for each channel
					{
						if (ik<10 || ik>BENG_CHANNELS-11) // only display data for first 10 and last 10 channels
						{
							int idx_beng = ik + ij*BENG_CHANNELS + ii*BENG_SNAPSHOTS*BENG_CHANNELS;
							if (il == 0)
							{
								beng_sample = beng_data_0[idx_beng];
							}
							else if (il == 1)
							{
								beng_sample = beng_data_1[idx_beng];
							}
							fprintf(stdout,"%2d+j%2d",(int)cuCrealf(beng_sample),(int)cuCimagf(beng_sample));
							if (ik < BENG_CHANNELS-1)
							{
								fprintf(stdout,", ");
							}
							if (ik == 9)
							{
								fprintf(stdout,"... , ");
							}
						}
					}
				}
				fprintf(stdout,"\n");
			}
		}
	}
	
	// Free device global memory
	#ifdef DEBUG
	printf("reader:DEBUG:Free device memory.\n");
	#endif
	cudaFree(gpu_vdif_buf);
	cudaFree(gpu_cid);
	cudaFree(gpu_fid);
	cudaFree(gpu_bcount);
	cudaFree(gpu_beng_data_0);
	cudaFree(gpu_beng_data_1);
	cudaFree(gpu_beng_frame_completion);
	// Free host memory
	#ifdef DEBUG
	printf("reader:DEBUG:Free host memory.\n");
	#endif
	free(vdif_buf);
	free(cid);
	free(fid);
	free(bcount);
	free(beng_data_0);
	free(beng_data_1);
	free(beng_frame_completion);
	
	// Reset the device and exit
	#ifdef DEBUG
	printf("reader:DEBUG:CUDA Device reset.\n");
	#endif
	cudaDeviceReset();
	
	#ifdef DEBUG
	printf("reader:DEBUG:Stop.\n");
	#endif
	
	if (logging)
	{
		fclose(fh_log);
	}
	exit(EXIT_SUCCESS);
}

/*
 * Parse VDIF frame and output data.
 * */
__global__ void vdif_reader(
	uint32_t *vdif_frames, 
	uint32_t *fid_out, 
	uint32_t *cid_out, 
	uint32_t *bcount_out, 
	cufftComplex *beng_data_out_0, 
	cufftComplex *beng_data_out_1, 
	uint32_t *beng_frame_completion,
	int32_t num_vdif_frames, 
	int blocks_per_grid)
{
	// VDIF header
	//~ __shared__ uint32_t vdif_header[VDIF_INT_SIZE_HEADER][THREADS_PER_BLOCK_Y];
	uint32_t cid,fid;
	uint32_t bcount; // we don't need very large bcount, just keep lower 32bits
	
	// VDIF data
	const uint32_t *vdif_frame_start; // pointer to start of the VDIF frame handled by this thread
	uint32_t samples_per_snapshot_half_0, samples_per_snapshot_half_1; // 4byte collections of 16 samples (2sums * (1real + 1imag) * 4xeng_parallel_chan) each
	float sample_real, sample_imag; // real and imaginary components from 2bit samples
	uint32_t idx_beng_data_out; // index into beng_data_out
	
	// misc
	uint32_t iframe; // VDIF frame sized index into VDIF data buffer currently processed by this thread
	int idata; // uint64_t sized index into VDIF data currently processed by this thread
	int isample; // 2bit sized index into the 8 consecutive channels of a single snapshot contained in each B-engine packet
	
	// control
	int old; // old value for B-engine completion counter
	
	/* iframe increases by the number of frames handled by a single grid.
	 * There are blocks_per_grid*THREADS_PER_BLOCK_Y frames handled simultaneously
	 * withing the grid.
	 * */
	for (iframe=0; iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y<num_vdif_frames; iframe+=blocks_per_grid*THREADS_PER_BLOCK_Y)
	{ 
		
		#ifdef DEBUG_GPU
			#ifdef DEBUG_GPU_CONDITION
				if ( DEBUG_GPU_CONDITION )
				{
			#endif // DEBUG_GPU_CONDITION
			printf("blk(thx,thy)=%d(%d,%d): #frame = %d + %d + %d*%d = %d < %d ? %s\n",blockIdx.x,threadIdx.x,threadIdx.y,
			iframe , threadIdx.y , blockIdx.x , THREADS_PER_BLOCK_Y,
			iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y,num_vdif_frames,
			iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y < num_vdif_frames ? "OK" : "NO");
			#ifdef DEBUG_GPU_CONDITION
				}
			#endif // DEBUG_GPU_CONDITION
		#endif // DEBUG_GPU
		
		//~ // This check has been moved into the iframe for statement
		//~ if (iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y < num_vdif_frames)
		//~ {
			/* Set the start of the VDIF frame handled by this thread. VDIF 
			 * frames are just linearly packed in memory. Consecutive y-threads
			 * read consecutive VDIF frames, and each x-block reads consecutive
			 * blocks of THREADS_PER_BLOCK_Y VDIF frames.
			 * */
			vdif_frame_start = vdif_frames + (iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y)*VDIF_INT_SIZE;
			
			//~ /* Read header data into shared buffer.
			 //~ * */
			//~ vdif_header[threadIdx.x][threadIdx.y] = *(vdif_frame_start+threadIdx.x);
			// B-engine c
			cid = (*(vdif_frame_start + BENG_VDIF_HDR_1_OFFSET_INT) & 0x000000FF);
			cid_out[iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y] = cid;
			// B-engine f
			fid = (*(vdif_frame_start + BENG_VDIF_HDR_1_OFFSET_INT) & 0x00FF0000)>>16;
			fid_out[iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y] = fid;
			// B-engine b
			bcount = ((*(vdif_frame_start + BENG_VDIF_HDR_1_OFFSET_INT)&0xFF000000)>>24) + ((*(vdif_frame_start + BENG_VDIF_HDR_0_OFFSET_INT)&0x00FFFFFF)<<8);
			bcount_out[iframe + threadIdx.y + blockIdx.x*THREADS_PER_BLOCK_Y] = bcount;
			
			/* Set the offset into the B-engine data buffer. Channels for 
			 * a single snapshot are consecutive in memory, consecutive 
			 * snapshots are separated by one spectrum, and consecutive
			 * B-engine frames are separated by 128 snapshots (128 spectra).
			 * */
			idx_beng_data_out  = BENG_CHANNELS*BENG_SNAPSHOTS*(bcount&BENG_BUFFER_INDEX_MASK); // offset given the masked B-engine counter value
			idx_beng_data_out += SWARM_XENG_PARALLEL_CHAN * (cid * SWARM_N_FIDS + fid); // offset given the cid and fid
			/* Add offset based on the threadIdx.x. Consecutive x-threads
			 * read consecutive 2-uint32_t (8byte) data chunks, which means
			 * that the target index for consecutive x-threads are separated
			 * as consecutive snapshots, i.e. single spectrum.
			 * */
			idx_beng_data_out += threadIdx.x*BENG_CHANNELS; // offset given the threadIdx.x
			
			/* idata increases by the number of uint32_t handled simultaneously
			 * by all x-threads. Each thread handles B-engine packet data 
			 * for a single snapshot per iteration.
			 * */
			for (idata=0; idata<VDIF_INT_SIZE_DATA; idata+=BENG_VDIF_INT_PER_SNAPSHOT*THREADS_PER_BLOCK_X)
			{
				/* Get sample data out of global memory. Offset from the 
				 * VDIF frame start by the header, the number of snapshots
				 * processed by the group of x-threads (idata), and the
				 * particular snapshot offset for THIS x-thread 
				 * (BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x).
				 * */
				samples_per_snapshot_half_0 = *(vdif_frame_start + VDIF_INT_SIZE_HEADER + idata + BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x);
				samples_per_snapshot_half_1 = *(vdif_frame_start + VDIF_INT_SIZE_HEADER + idata + BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x + 1);
				for (isample=0; isample<SWARM_XENG_PARALLEL_CHAN/2; isample++)
				{
					// channels a-d
					sample_real = (float)(samples_per_snapshot_half_0 & 0x03); // phased sum 0, real: b1b0
					samples_per_snapshot_half_0 = samples_per_snapshot_half_0 >> VDIF_BIT_DEPTH;
					sample_imag = (float)(samples_per_snapshot_half_0 & 0x03); // phased sum 0, imag: b3b2
					samples_per_snapshot_half_0 = samples_per_snapshot_half_0 >> VDIF_BIT_DEPTH;
					beng_data_out_0[idx_beng_data_out] = make_cuFloatComplex(sample_real, sample_imag);
					sample_real = (float)(samples_per_snapshot_half_0 & 0x03); // phased sum 1, real: b5b4
					samples_per_snapshot_half_0 = samples_per_snapshot_half_0 >> VDIF_BIT_DEPTH;
					sample_imag = (float)(samples_per_snapshot_half_0 & 0x03); // phased sum 1, imag: b7b6
					samples_per_snapshot_half_0 = samples_per_snapshot_half_0 >> VDIF_BIT_DEPTH;
					beng_data_out_1[idx_beng_data_out] = make_cuFloatComplex(sample_real, sample_imag);
					
					// channels e-h
					sample_real = (float)(samples_per_snapshot_half_1 & 0x03); // phased sum 0, real: b1b0
					samples_per_snapshot_half_1 = samples_per_snapshot_half_1 >> VDIF_BIT_DEPTH;
					sample_imag = (float)(samples_per_snapshot_half_1 & 0x03); // phased sum 0, imag: b3b2
					samples_per_snapshot_half_1 = samples_per_snapshot_half_1 >> VDIF_BIT_DEPTH;
					beng_data_out_0[idx_beng_data_out+SWARM_XENG_PARALLEL_CHAN/2] = make_cuFloatComplex(sample_real, sample_imag);
					sample_real = (float)(samples_per_snapshot_half_1 & 0x03); // phased sum 1, real: b5b4
					samples_per_snapshot_half_1 = samples_per_snapshot_half_1 >> VDIF_BIT_DEPTH;
					sample_imag = (float)(samples_per_snapshot_half_1 & 0x03); // phased sum 1, imag: b7b6
					samples_per_snapshot_half_1 = samples_per_snapshot_half_1 >> VDIF_BIT_DEPTH;
					beng_data_out_1[idx_beng_data_out+SWARM_XENG_PARALLEL_CHAN/2] = make_cuFloatComplex(sample_real, sample_imag);
					
					idx_beng_data_out++; // next channel a,e -> b,f etc
				} // for (isample=0; ...)
				/* The next snapshot handled by this thread will increment
				 * by the number of x-threads, so index into B-engine data
				 * should increment by that many spectra.
				 * */
				idx_beng_data_out += THREADS_PER_BLOCK_X*BENG_CHANNELS;
			} // for (idata=0; ...)
			
			#ifdef DEBUG_GPU
				#ifdef DEBUG_GPU_CONDITION
					if ( DEBUG_GPU_CONDITION )
					{
				#endif // DEBUG_GPU_CONDITION
				printf("blk(thx,thy)=%d(%d,%d): cid=%3d, fid=%d, bcount=%8d (masked=%3d); idx_beng_data_out = %d*%d*%d + %d*(%3d*%d+%d) + %d*%d =  %9d + %9d + %9d = %9d --> %9d.\n",
					blockIdx.x,threadIdx.x,threadIdx.y,cid,fid,bcount,bcount & BENG_BUFFER_INDEX_MASK,
					BENG_CHANNELS,BENG_SNAPSHOTS,(bcount & BENG_BUFFER_INDEX_MASK),
					SWARM_XENG_PARALLEL_CHAN , cid , SWARM_N_FIDS , fid,
					threadIdx.x,BENG_CHANNELS,
					BENG_CHANNELS*BENG_SNAPSHOTS*(bcount & BENG_BUFFER_INDEX_MASK),
					SWARM_XENG_PARALLEL_CHAN * (cid * SWARM_N_FIDS + fid),
					threadIdx.x*BENG_CHANNELS,
					BENG_CHANNELS*BENG_SNAPSHOTS*(bcount & BENG_BUFFER_INDEX_MASK)+SWARM_XENG_PARALLEL_CHAN * (cid * SWARM_N_FIDS + fid)+threadIdx.x*BENG_CHANNELS,
					idx_beng_data_out);
				#ifdef DEBUG_GPU_CONDITION
					}
				#endif // DEBUG_GPU_CONDITION
			#endif // DEBUG_GPU
			
			// reset completion counter for two B-engine frames behind
			beng_frame_completion[(bcount-2)&BENG_BUFFER_INDEX_MASK] = 0;
			
			// increment completion counter for this B-engine frame
			old = atomicAdd(beng_frame_completion + (bcount&BENG_BUFFER_INDEX_MASK), 1);
			#ifdef DEBUG_GPU
				#ifdef DEBUG_GPU_CONDITION
					if ( DEBUG_GPU_CONDITION )
					{
				#endif // DEBUG_GPU_CONDITION
				printf("blk(thx,thy)=%d(%d,%d): B-engine frame bcount=%8d (masked=%3d) completion increment: %6d --> %6d (FULL = %6d).\n",
					blockIdx.x,threadIdx.x,threadIdx.y,bcount,bcount & BENG_BUFFER_INDEX_MASK,old,old+1,BENG_FRAME_COMPLETION_COMPLETE);
				#ifdef DEBUG_GPU_CONDITION
					}
				#endif // DEBUG_GPU_CONDITION
			#endif // DEBUG_GPU
			
			/* Vote to see if the frame is complete. This will be indicated
			 * by the old value of the counter being one less than what indicates
			 * a full frame in one of the threads.
			 * */
			if (__any(old == BENG_FRAME_COMPLETION_COMPLETE-1))
			{
				// do something...
				#ifdef DEBUG_GPU
					#ifdef DEBUG_GPU_CONDITION
						if ( DEBUG_GPU_CONDITION )
						{
					#endif // DEBUG_GPU_CONDITION
					printf("blk(thx,thy)=%d(%d,%d): B-engine frame bcount=%8d (masked=%3d) complete.\n",
						blockIdx.x,threadIdx.x,threadIdx.y,bcount,bcount & BENG_BUFFER_INDEX_MASK);
					#ifdef DEBUG_GPU_CONDITION
						}
					#endif // DEBUG_GPU_CONDITION
				#endif // DEBUG_GPU
			}
		//~ } // if (iframe + ... < num_vdif_frames)
	} // for (iframe=0; ...)
}

/*
 * CUDA error code checker.
 * 
 * Tests whether the CUDA error code returned is an error or success. In
 * case of error a message is displayed and the program exits.
 */
void error_check(void)
{
	cudaError_t err = cudaGetLastError();
	error_check(err);
}
void error_check(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
