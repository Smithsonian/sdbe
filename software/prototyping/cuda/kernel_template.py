kernel_template = """
/** @file kernel_template.py
@author Katherine Rosenfeld
@author Andre Young
@version 0.1
@brief GPU kernels for pipeline components.
  
@details GPU kernels for the following pipeline components:
- VDIF interpreter
- B-engine depacketizer
- Pre-preprocessor
- Re-ordering
- Linear interpolation
- Nearest-neighbor interpolation
- 2bit quantization

@date June 2015
*/

/**
@mainpage
These CUDA kernels resample 2496 MHz SWARM B-engine packets to
match the 4096 MHz R2DBE.  The output is two separate timestreams
which should be band-trimmed and frequency shifted using cufft.
*/

#include <cufft.h>

#define INTERPOLATION_FACTOR 64
#define DECIMATION_FACTOR 39

// VDIF constants
#define VDIF_BYTE_SIZE 1056 // VDIF frame size in bytes
#define VDIF_BYTE_SIZE_HEADER 32 // VDIF header size in bytes
#define VDIF_BYTE_SIZE_DATA 1024 // VDIF data size in bytes
#define VDIF_INT_SIZE (1056/4) // VDIF frame size in int
#define VDIF_INT_SIZE_HEADER (32/4) // VDIF header size in int
#define VDIF_INT_SIZE_DATA (1024/4) // VDIF data size in int
#define VDIF_BIT_DEPTH 2 // bits-per-sample

// Data structure
#define BENG_CHANNELS_ 16384
#define BENG_CHANNELS (BENG_CHANNELS_+1) // number of channels PLUS added sample-rate/2 component for the complex-to-real inverse transform
#define BENG_SNAPSHOTS 128
#define BENG_BUFFER_IN_COUNTS %(BENG_BUFFER_IN_COUNTS)d
#define SWARM_N_FIDS 8
#define SWARM_XENG_PARALLEL_CHAN 8
#define BENG_FRAMES_OUT_CONSECUTIVE_SNAPSHOTS // if defined, B-engine frames are stored such that the same spectral channel for consecutive snapshots are adjacent in memory

// VDIF packed B-engine packet
#define BENG_VDIF_HDR_0_OFFSET_INT 4 // b1 b2 b3 b4
#define BENG_VDIF_HDR_1_OFFSET_INT 5 //  c  0  f b0
#define BENG_VDIF_CHANNELS_PER_INT 4 // a-d in a single int32_t, and e-h in a single int32_t
#define BENG_VDIF_INT_PER_SNAPSHOT (SWARM_XENG_PARALLEL_CHAN/BENG_VDIF_CHANNELS_PER_INT)
#define BENG_PACKETS_PER_FRAME (BENG_CHANNELS_/SWARM_XENG_PARALLEL_CHAN)
#define BENG_FRAME_COMPLETION_COMPLETE_ON_GPU (BENG_PACKETS_PER_FRAME*blockDim.x) // value of completion counter when B-engine frame complete, multiplication by THREADS_PER_BLOCK_x required since all x-threads increment counter
#define BENG_FRAME_COMPLETION_COMPLETE_ON_CPU (BENG_PACKETS_PER_FRAME*num_x_threads) // value of completion counter when B-engine frame complete, multiplication by THREADS_PER_BLOCK_x required since all x-threads increment counter
#define BENG_VDIF_SAMPLE_VALUE_OFFSET 2.0f

// Output 
#define OUTPUT_BITS_PER_SAMPLE 2
#define OUTPUT_MAX_VALUE_MASK ((int)pow((double) 2,OUTPUT_BITS_PER_SAMPLE) - 1)
#define OUTPUT_SAMPLES_PER_INT (32/OUTPUT_BITS_PER_SAMPLE)

/**
Read B-engine C-stamp from VDIF header.
@author Andre Young
*/
__device__ int32_t get_cid_from_vdif(const int32_t *vdif_start){
  return (*(vdif_start + BENG_VDIF_HDR_1_OFFSET_INT) & 0x000000FF);
}

/**
Read B-engine F-stamp from VDIF header
@author Andre Young
*/
__device__ int32_t get_fid_from_vdif(const int32_t *vdif_start){
  return (*(vdif_start + BENG_VDIF_HDR_1_OFFSET_INT) & 0x00FF0000)>>16;
}

/**
Read B-engine B-counter from VDIF header
@author Andre Young
*/
__device__ int32_t get_bcount_from_vdif(const int32_t *vdif_start){
  return ((*(vdif_start + BENG_VDIF_HDR_1_OFFSET_INT)&0xFF000000)>>24) + ((*(vdif_start + BENG_VDIF_HDR_0_OFFSET_INT)&0x00FFFFFF)<<8);
}

/**
Read complex sample pair and shift input data accordingly inplace.
@author Andre Young
*/
__device__ cufftComplex read_complex_sample(int32_t *samples_int)
{
 float sample_imag, sample_real;
  sample_imag = __int2float_rd(*samples_int & 0x03) - 2.0f;
 *samples_int = (*samples_int) >> 2;
  sample_real = __int2float_rd(*samples_int & 0x03) - 2.0f;
 *samples_int = (*samples_int) >> 2;
 return make_cuFloatComplex(sample_real, sample_imag);
}

/** @brief Parse VDIF frame and store B-engine frames in buffer
Parses SWARM SDBE VDIF frames and stores de-quantized B-engine frames
in two buffers (one for each time-stream).  Before use, the data
must be reshuffled.
@param[in]  vdif_frames holds the 32bit word VDIF frames
@param[out] beng_data_out_0 is the LSB phased sum (8150 -- 6902 MHz)
@param[out] beng_data_out_1 is the USB phased sum (7850 -- 9098 MHz)
@author Andre Young
@date June 2015
*/
__global__ void vdif_to_beng(
 int32_t *vdif_frames,
 int32_t *fid_out,
 int32_t *cid_out,
 int32_t *bcount_out,
 cufftComplex *beng_data_out_0,
 cufftComplex *beng_data_out_1,
 int32_t *beng_frame_completion,
 int32_t num_vdif_frames,
 int32_t bcount_offset){

  int32_t cid,fid;
  int32_t bcount;
  const int32_t *vdif_frame_start;
  int32_t samples_per_snapshot_half_0, samples_per_snapshot_half_1;
  int32_t idx_beng_data_out;
  int32_t iframe;
  int idata;
  int isample;
  //int old;

  /* iframe increases by the number of frames handled by a single grid.
   * There are gridDim.x*gridDim.y*blockDim.y frames handled simultaneously
   * within the grid.
   */
  for (iframe=0; iframe + threadIdx.y + blockIdx.x*blockDim.y<num_vdif_frames; iframe+=gridDim.x*gridDim.y*blockDim.y){
   /* Set the start of the VDIF frame handled by this thread. VDIF 
    * frames are just linearly packed in memory. Consecutive y-threads
    * read consecutive VDIF frames, and each x-block reads consecutive
    * blocks of blockDim.y VDIF frames.
    */
    vdif_frame_start = vdif_frames + (iframe + threadIdx.y + blockIdx.x*blockDim.y)*(VDIF_INT_SIZE);
    cid = get_cid_from_vdif(vdif_frame_start);
    fid = get_fid_from_vdif(vdif_frame_start);
    bcount = get_bcount_from_vdif(vdif_frame_start);
    cid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = cid;
    fid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = fid;
    bcount_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = bcount;

    /* Reorder to have snapshots contiguous and consecutive channels
     * separated by 128 snapshots times the number of B-engine frames
     * in buffer. This means consecutive x-threads will handle consecutive snapshots.
     */
    idx_beng_data_out = SWARM_XENG_PARALLEL_CHAN * (cid * SWARM_N_FIDS + fid)*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS;
    idx_beng_data_out += ((bcount-bcount_offset) %% BENG_BUFFER_IN_COUNTS)*BENG_SNAPSHOTS;
    idx_beng_data_out += threadIdx.x;

    /* idata increases by the number of int32_t handled simultaneously
     * by all x-threads. Each thread handles B-engine packet data 
     * for a single snapshot per iteration.
     */
    for (idata=0; idata<VDIF_INT_SIZE_DATA; idata+=BENG_VDIF_INT_PER_SNAPSHOT*blockDim.x){
      /* Get sample data out of global memory. Offset from the 
       * VDIF frame start by the header, the number of snapshots
       * processed by the group of x-threads (idata), and the
       * particular snapshot offset for THIS x-thread 
       * (BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x).
       */
      samples_per_snapshot_half_0 = *(vdif_frame_start + VDIF_INT_SIZE_HEADER + idata + BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x);
      samples_per_snapshot_half_1 = *(vdif_frame_start + VDIF_INT_SIZE_HEADER + idata + BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x + 1);

      for (isample=0; isample<SWARM_XENG_PARALLEL_CHAN/2; ++isample){
        beng_data_out_1[idx_beng_data_out+(SWARM_XENG_PARALLEL_CHAN/2-(isample+1))*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS] = read_complex_sample(&samples_per_snapshot_half_0);
        beng_data_out_0[idx_beng_data_out+(SWARM_XENG_PARALLEL_CHAN/2-(isample+1))*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS] = read_complex_sample(&samples_per_snapshot_half_0);
        beng_data_out_1[idx_beng_data_out+(SWARM_XENG_PARALLEL_CHAN/2-(isample+1)+SWARM_XENG_PARALLEL_CHAN/2)*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS] = read_complex_sample(&samples_per_snapshot_half_1);
        beng_data_out_0[idx_beng_data_out+(SWARM_XENG_PARALLEL_CHAN/2-(isample+1)+SWARM_XENG_PARALLEL_CHAN/2)*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS] = read_complex_sample(&samples_per_snapshot_half_1);
      }

      /* The next snapshot handled by this thread will increment
      * by the number of x-threads, so index into B-engine data
      * should increment by that number.
      */
      idx_beng_data_out += blockDim.x;
    } // for (idata=0; ...)

    // increment completion counter for this B-engine frame
    //old = atomicAdd(beng_frame_completion + ((bcount-bcount_offset) %% BENG_BUFFER_IN_COUNTS), 1);

    /* Vote to see if the frame is complete. This will be indicated
     * by the old value of the counter being one less than what indicates
     * a full frame in one of the threads.
     */
    //if (__any(old == BENG_FRAME_COMPLETION_COMPLETE_ON_GPU-1)){
        //do something...
    //}
  } // for (iframe=0; ...)
}


/**@brief Reorder B-engine data for BENG_FRAMES_OUT_CONSECUTIVE_SNAPSHOTS
Reorder B-engine data that is snapshot contiguous and where consecutive channels
are separated by 128 snapshots time the number of B-engine frames in the buffer.
This kernel must be called with a (16,16) thread block and enough x-blocks
to cover 1 B-engine.
- blockDim.x = 16
- blockDim.y = 16;
- gridDim.x = BENG_CHANNELS_ * BENG_SNAPSHOTS / (blockDim.x*blockDim.y)
@author Katherine Rosenfeld
@date June 2015
*/
__global__ void reorderTz_smem(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){
  // gridDim.x = 16384 * 128 / (16 * 16) = 8192
  // blockDim.x = 16; blockDim.y = 16;
  // --> launches 2097152 threads

  int32_t sid_out,bid_in;

  __shared__ cufftComplex tile[16][16];

  // for now, let us loop the grid over B-engine frames:
  for (int bid_out=0; bid_out<num_beng_frames-1; bid_out+=1){

    // input snapshot id
    int sid_in = (blockIdx.x * blockDim.x + threadIdx.x) & (BENG_SNAPSHOTS-1);
    // input channel id 
    int cid = threadIdx.y + blockDim.y * (blockIdx.x / (128 / blockDim.x));

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    // and by 1-B-frame:
    if (sid_out < 69){
      bid_in = bid_out;
    } else {
      bid_in = bid_out+1;
    }

    tile[threadIdx.x][threadIdx.y] = beng_data_in[BENG_SNAPSHOTS*num_beng_frames*cid + BENG_SNAPSHOTS*bid_in + sid_in];

    __syncthreads();

    // now we transpose warp orientation over channels and snapshot index

    // snapshot id
    sid_in = threadIdx.y + (blockIdx.x*blockDim.y) & (BENG_SNAPSHOTS-1);
    // channel id 
    cid = threadIdx.x + blockDim.x * (blockIdx.x / (BENG_SNAPSHOTS / blockDim.x)); 

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    beng_data_out[BENG_SNAPSHOTS*BENG_CHANNELS*bid_out + BENG_CHANNELS*sid_out + cid] = tile[threadIdx.y][threadIdx.x];

    // zero out nyquist: 
    if (cid == 0) {
      beng_data_out[BENG_SNAPSHOTS*BENG_CHANNELS*bid_out + BENG_CHANNELS*sid_out + BENG_CHANNELS_] = make_cuComplex(0.,0.);
    }

    __syncthreads();
  }
}

/** @brief Nearest-neighbor interpolation kernel
This kernel uses a round-half-to-even tie-breaking rule for
nearest-neighbor interpolation.  This is opposite that of 
python's interp_1d.
@author Katherine Rosenfeld
@author June 2015
*/
__global__ void nearest(float *a, int Na, float *b, int Nb, double c){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < Nb) {
    int ida = __double2int_rn(tid*c); // round nearest
      b[tid] = a[ida];
  }
}

/** @brief Linear interpolation kernel
This kernel performs a linear interpolation assuming that the weights and input indices have been precomputed. 
These values are computed for chunks of DOWNSAMPLING_FACTOR snapshots and the
x-threads should cover one chunk of output R2DBE data.
@author Katherine Rosenfeld
@date June 2015 
*/
__global__ void linear(const float *a, float *b, const int32_t N, const float *wgt, const int *ida){
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t i = ida[tid];
  float w = wgt[tid];
  // loop over groups of DECIMATION_FACTOR snapshots
  for (int32_t ichunk=0; ichunk<(BENG_BUFFER_IN_COUNTS-1)*BENG_SNAPSHOTS/DECIMATION_FACTOR; ichunk += 1){
    if (i+1+2*BENG_CHANNELS_*DECIMATION_FACTOR*ichunk<N){
      b[ichunk*2*BENG_CHANNELS_*INTERPOLATION_FACTOR+tid] = 
				a[i+2*BENG_CHANNELS_*DECIMATION_FACTOR*ichunk]*(1.0f-w) + 
			        a[i+1+2*BENG_CHANNELS_*DECIMATION_FACTOR*ichunk]*w;
    } else {
      b[ichunk*2*BENG_CHANNELS_*INTERPOLATION_FACTOR+tid] = 
				a[i+2*BENG_CHANNELS_*DECIMATION_FACTOR*ichunk];
    }
  }
}

__global__ void zero_cout(cufftComplex *a, const int32_t n)
{
  /* Zero out an complex array */
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n){
    a[tid] = make_cuComplex(0.,0.);
  }
}

__global__ void zero_rout(float *a, const int32_t n)
{
  /* Zero out a floating point array */
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n){
    a[tid] = 0.;
  }
}

__global__ void strided_copy(float *a, int istart, float *b, int N, int istride, int iskip)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < N){
    b[tid] = a[istart+(tid/istride)*(istride+iskip) + (tid %% istride) ];
  }
}

/** @brief 2-bit quantization kernel

This 2bit quantization kernel must be called with 16 x-threads,
any number of y-threads, and any number of x-blocks.
@author Andre Young
@date June 2015
*/
__global__ void quantize2bit(const float *in, unsigned int *out, int N, float thresh)
{
	int idx_in = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int idx_out = blockIdx.x*blockDim.y + threadIdx.y;
	
	for (int ii=0; (idx_in+ii)<N; ii+=gridDim.x*blockDim.x*blockDim.y)
	{
		//This is is for 00 = -2, 01 = -1, 10 = 0, 11 = 1.
		/* Assume sample x > 0, lower bit indicates above threshold. Then
		 * test, if x < 0, XOR with 11.
		 */
		//int sample_2bit = ( ((fabsf(in[idx_in+ii]) >= thresh) | 0x02) ^ (0x03*(in[idx_in+ii] < 0)) ) & 0x3;
		int sample_2bit = ( ((fabsf(in[idx_in+ii]) >= thresh) | 0x02) ^ (0x03*(in[idx_in+ii] < 0)) ) & OUTPUT_MAX_VALUE_MASK;
		//~ //This is for 11 = -2, 10 = -1, 01 = 0, 10 = 1
		//~ int sample_2bit = ((fabsf(in[idx_in+ii]) <= thresh) | ((in[idx_in+ii] < 0)<<1)) & OUTPUT_MAX_VALUE_MASK;
		sample_2bit = sample_2bit << (threadIdx.x*2);
		atomicOr(out+idx_out, sample_2bit);
		idx_out += gridDim.x*blockDim.y;
	}
}
"""
