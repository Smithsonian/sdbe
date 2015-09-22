extern "C" {

#include <stdio.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/time.h>

#include "aphids.h"
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"
#include "vdif_out_databuf.h"
//~ #include "vdif_upload_to_gpu_databuf.h"
#include "vdif_inout_gpu_thread.h"
}

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverSp.h>

//#define GPU_DEBUG		// slows down performance
#define GPU_COMPUTE
//#define GPU_MULTI
#define QUANTIZE_THRESHOLD_COMPUTE // toggle threshold calculation
//#define QUANTIZE_THRESHOLD 1.f

#ifdef GPU_MULTI
#define NUM_GPU 4
#else
#define NUM_GPU 1
#endif

#define STATE_ERROR    -1
#define STATE_INIT      0
#define STATE_PROC      1

#ifdef GPU_DEBUG
void reportDeviceMemInfo(void){
  size_t avail,total;
  cudaMemGetInfo(&avail,&total);
  fprintf(stdout, "GPU_DEBUG : Dev mem (avail, tot, frac): %u , %u  = %0.4f\n", 
		(unsigned)avail, (unsigned)total, 100.f*avail/total);
}
#endif

static void HandleError( cudaError_t err,const char *file,int line ) {
#ifdef GPU_DEBUG
    if (err != cudaSuccess) {
        fprintf(stderr,"%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
    }
#endif
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(x));\
    return EXIT_FAILURE;}} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUSPARSE_CALL(x) do { if((x)!=CUSPARSE_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define PI 3.14159265359
__host__ __device__ float hamming(int n, int m){
  return 0.54 - 0.46*cos(2.*PI*n/(m-1.));
}

/**
 * structure used to accumulate the moments and other 
 * statistical properties encountered so far.
 * @author: Joseph Rhoads
 * reference: https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu
 * */
template <typename T>
struct summary_stats_data
{
    T n;
    T mean;
    T M2;

    // initialize to the identity element
    void initialize()
    {
      n = mean = M2 = 0;
    }

    T average()       { return mean; }
    T variance()   { return M2 / (n - 1); }
    T variance_n() { return M2 / n; }
};

/**
 * stats_unary_op is a functor that takes in a value x and
 * returns a variace_data whose mean value is initialized to x.
 * @author: Joseph Rhoads
 * reference: https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu
 * */
template <typename T>
struct summary_stats_unary_op
{
    __host__ __device__
    summary_stats_data<T> operator()(const T& x) const
    {
         summary_stats_data<T> result;
         result.n    = 1;
         result.mean = x;
         result.M2   = 0;

         return result;
    }
};

/**
 * summary_stats_binary_op is a functor that accepts two summary_stats_data 
 * structs and returns a new summary_stats_data which are an
 * approximation to the summary_stats for 
 * all values that have been agregated so far
 * @author: Joseph Rhoads
 * reference: https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu
 * */
template <typename T>
struct summary_stats_binary_op 
    : public thrust::binary_function<const summary_stats_data<T>&, 
                                     const summary_stats_data<T>&,
                                           summary_stats_data<T> >
{
    __host__ __device__
    summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
    {
        summary_stats_data<T> result;

        // precompute some common subexpressions
        T n  = x.n + y.n;
        T delta  = y.mean - x.mean;
        T delta2 = delta  * delta;

        //Basic number of samples (n)
        result.n   = n;

        result.mean = x.mean + delta * y.n / n;
        result.M2  = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;
        return result;
    }
};

/**
 * Structure for APHIDS resampling 
 * @author Katherine Rosenfeld
 * */

typedef struct aphids_resampler {
    int deviceId;
    int skip_chan;
    float quantizeThreshold_0, quantizeThreshold_1;
    float quantizeOffset_0, quantizeOffset_1;
    summary_stats_data<float> ssd;
    cufftComplex *gpu_A_0, *gpu_A_1;
    cufftComplex *gpu_B_0, *gpu_B_1;
    vdif_out_data_group_t *gpu_out_buf;
    int fft_size[3],batch[3],repeat[3];
    cufftHandle cufft_plan[3];
#ifdef GPU_DEBUG
    cudaStream_t stream;
    cudaEvent_t tic,toc;
#endif

} aphids_resampler_t;

/** @brief Initialize resampler structure (including device memory).
 */
int aphids_resampler_init(aphids_resampler_t *resampler, int _deviceId) {

  // create FFT plans
  int inembed[3], onembed[3];
  cufftResult cufft_status = CUFFT_SUCCESS;

  // device ID
  resampler->deviceId = _deviceId;

  // quantization thresholds:
  resampler->quantizeThreshold_0 = 0;
  resampler->quantizeThreshold_1 = 0;
  // quantization offsets:
  resampler->quantizeOffset_0 = 0;
  resampler->quantizeOffset_1 = 0;

  // switch to device
  cudaSetDevice(resampler->deviceId);

#ifdef GPU_DEBUG
  int i;
  size_t workSize[1];
  cudaStreamCreate(&(resampler->stream));
  cudaEventCreate(&(resampler->tic));
  cudaEventCreate(&(resampler->toc));
#endif // GPU_DEBUG

  // allocate device memory
  cudaMalloc((void **)&(resampler->gpu_A_0), BENG_FRAMES_PER_GROUP*BENG_SNAPSHOTS*BENG_CHANNELS_*sizeof(cufftComplex));	// 671088640B
  cudaMalloc((void **)&(resampler->gpu_A_1), BENG_FRAMES_PER_GROUP*BENG_SNAPSHOTS*BENG_CHANNELS_*sizeof(cufftComplex));	// 671088640B
  cudaMalloc((void **)&(resampler->gpu_B_0), (BENG_FRAMES_PER_GROUP-1)*BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS*sizeof(cufftComplex));	// 654950400B
  cudaMalloc((void **)&(resampler->gpu_B_1), (BENG_FRAMES_PER_GROUP-1)*BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS*sizeof(cufftComplex));	// 654950400B
  cudaMalloc((void **)&(resampler->gpu_out_buf), sizeof(vdif_out_data_group_t)); // 67108864B

 /*
 * http://docs.nvidia.com/cuda/cufft/index.html#cufft-setup
 * iFFT transforming complex SWARM spectra into real time series.
 * Input data is padded according to UNPACKED_BENG_CHANNELS >= BENG_CHANNELS.
 * Single batch size is BENG_SNAPHOTS and so to cover the entire databuf
 * this must repeated BENG_FRAMES_PER_GROUP-1 times (saves 14.8% memory)
 */
  resampler->fft_size[0] = 2*BENG_CHANNELS_;
  inembed[0]  = UNPACKED_BENG_CHANNELS; 
  onembed[0]  = 2*BENG_CHANNELS_;
  resampler->batch[0]    = BENG_SNAPSHOTS;
  resampler->repeat[0]   = BENG_FRAMES_PER_GROUP - 1;
  cufft_status = cufftPlanMany(&(resampler->cufft_plan[0]), 1, &(resampler->fft_size[0]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_C2R,resampler->batch[0]);
  if (cufft_status != CUFFT_SUCCESS)
  {
    hashpipe_error(__FILE__, "CUFFT error: plan 0 creation failed");
    //state = STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cufftGetSize(resampler->cufft_plan[0],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 0 is %dx%dx%d\n", resampler->repeat[0],resampler->batch[0],resampler->fft_size[0]);
  fprintf(stdout,"GPU_DEBUG : plan 0 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  

  /*
 * FFT transforming time series into complex spectrum.
 * Input data has dimension RESAMPLING_CHUNK_SIZE with
 * Set the batch size to be RESAMPLING_BATCH with
 * (BENG_FRAMES_PER_GROUP-1)*2*BENG_CHANNELS_*BENG_SNAPSHOTS/RESAMPLING_CHUNK_SIZE / RESAMPLING_BATCH
 *  required iterations.
 */
  resampler->fft_size[1] = RESAMPLING_CHUNK_SIZE;
  inembed[0]  = RESAMPLING_CHUNK_SIZE; 
  onembed[0]  = RESAMPLING_CHUNK_SIZE/2+1;
  resampler->batch[1]  = RESAMPLING_BATCH;
  resampler->repeat[1] = (BENG_FRAMES_PER_GROUP-1)*2*BENG_CHANNELS_*BENG_SNAPSHOTS/RESAMPLING_CHUNK_SIZE/RESAMPLING_BATCH;
  if( cufftPlanMany(&(resampler->cufft_plan[1]), 1, &(resampler->fft_size[1]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_R2C,resampler->batch[1]) != CUFFT_SUCCESS)
  {
    hashpipe_error(__FILE__, "CUFFT error: plan 1 creation failed");
    //state = STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cufftGetSize(resampler->cufft_plan[1],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 1 is %dx%dx%d\n", resampler->repeat[1],resampler->batch[1],resampler->fft_size[1]);
  fprintf(stdout,"GPU_DEBUG : plan 1 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  

 /*
 * FFT transforming complex spectrum into resampled time series simultaneously
 * trimming the SWARM guard bands (150 MHz -- 1174 MHz).
 * Input data has dimension EXPANSIONS_FACTOR
 */
  resampler->skip_chan = (int)(150e6 / (SWARM_RATE / RESAMPLING_CHUNK_SIZE));
  resampler->fft_size[2] = RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR;
  inembed[0]  = RESAMPLING_CHUNK_SIZE/2+1; 
  onembed[0]  = RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR;
  resampler->batch[2]    = resampler->batch[1];
  resampler->repeat[2]   = resampler->repeat[1];
  if( cufftPlanMany(&(resampler->cufft_plan[2]), 1, &(resampler->fft_size[2]),
		inembed,1,inembed[0],
		onembed,1,onembed[0],
		CUFFT_C2R,resampler->batch[2]) != CUFFT_SUCCESS )
  {
    hashpipe_error(__FILE__, "CUFFT error: plan 2 creation failed");
    //state = STATE_ERROR;
  }


#ifdef GPU_DEBUG
  for (i=0; i<3; ++i){
    cufftSetStream(resampler->cufft_plan[i], resampler->stream);
  }
  cufftGetSize(resampler->cufft_plan[2],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 2 is %dx%dx%d\n", resampler->repeat[2],resampler->batch[2],resampler->fft_size[2]);
  fprintf(stdout,"GPU_DEBUG : masking first %d channels \n",resampler->skip_chan);
  fprintf(stdout,"GPU_DEBUG : plan 2 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  

  return 1;
}

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
__device__ uint64_t get_bcount_from_vdif(const int32_t *vdif_start){
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
// int32_t *fid_out,
// int32_t *cid_out,
// int32_t *bcount_out,
 cufftComplex *beng_data_out_0,
 cufftComplex *beng_data_out_1,
// int32_t *beng_frame_completion,
 int32_t num_vdif_frames,
 uint64_t b_zero){
// int32_t bcount_offset){
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
    bcount = (int32_t)(get_bcount_from_vdif(vdif_frame_start) - b_zero);
    //cid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = cid;
    //fid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = fid;
    //bcount_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = bcount;
    /* Reorder to have snapshots contiguous and consecutive channels
     * separated by 128 snapshots times the number of B-engine frames
     * in buffer. This means consecutive x-threads will handle consecutive snapshots.
     */
    idx_beng_data_out = SWARM_XENG_PARALLEL_CHAN * (cid * SWARM_N_FIDS + fid)*BENG_BUFFER_IN_COUNTS*BENG_SNAPSHOTS;
    //idx_beng_data_out += ((bcount-bcount_offset) % BENG_BUFFER_IN_COUNTS)*BENG_SNAPSHOTS;
    idx_beng_data_out += (bcount % BENG_BUFFER_IN_COUNTS)*BENG_SNAPSHOTS; // bcount_offset = 0
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
    //old = atomicAdd(beng_frame_completion + ((bcount-bcount_offset) % BENG_BUFFER_IN_COUNTS), 1);
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
to cover 1 B-engine.  The output has 16400 zero-padded channels
- blockDim.x = 16
- blockDim.y = 16;
- gridDim.x = BENG_CHANNELS_ * BENG_SNAPSHOTS / (blockDim.x*blockDim.y)
@author Katherine Rosenfeld
@date June 2015
*/
__global__ void reorderTzp_smem(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){
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
    int cid = threadIdx.y + blockDim.y * (blockIdx.x / (BENG_SNAPSHOTS / blockDim.x));

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

    beng_data_out[(BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS)*bid_out + UNPACKED_BENG_CHANNELS*sid_out + cid] 
	= tile[threadIdx.y][threadIdx.x];

    // zero out nyquist: 
    if (cid < 16) {
      beng_data_out[(BENG_SNAPSHOTS*UNPACKED_BENG_CHANNELS)*bid_out + UNPACKED_BENG_CHANNELS*sid_out + BENG_CHANNELS_ + cid]	= make_cuComplex(0.,0.);
    }

    __syncthreads();
  }
}

/** @brief 2-bit quantization kernel

This 2bit quantization kernel must be called with 16 x-threads,
any number of y-threads, and any number of x-blocks.
@author Andre Young
@date June 2015
*/
__global__ void quantize2bit(const float *in, unsigned int *out, int N, float thresh, float offset)
{
	int idx_in = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int idx_out = blockIdx.x*blockDim.y + threadIdx.y;
	
	for (int ii=0; (idx_in+ii)<N; ii+=gridDim.x*blockDim.x*blockDim.y)
	{
		float val_in = in[idx_in+ii] - offset;
		//This is is for 00 = -2, 01 = -1, 10 = 0, 11 = 1.
		/* Assume sample x > 0, lower bit indicates above threshold. Then
		 * test, if x < 0, XOR with 11.
		 */
		int sample_2bit = ( ((fabsf(val_in) >= thresh) | 0x02) ^ (0x03*(val_in < 0)) ) & OUTPUT_MAX_VALUE_MASK;
		sample_2bit = sample_2bit << (threadIdx.x*2);
		out[idx_out] = 0;
		atomicOr(out+idx_out, sample_2bit);
		idx_out += gridDim.x*blockDim.y;
	}
}

/*
 d_s is complex PFB timestream [num_snapshots, num_freqs]
*/
int inverse_pfb(cufftComplex *d_s, int num_samples, int num_tap, int num_freq, float *d_rts){
  // pull out the number of blocks and their length
  int lblock = 2 * (num_freq - 1);
  int nblock = num_samples / lblock - (num_tap - 1);
  int ntsblock = nblock + num_tap - 1;

  cublasHandle_t cublasH;
  cusparseHandle_t cusparseH;
  cufftHandle plan;
  float *d_pts, *d_yh;
  const float alpha = 1., beta = 0.;

  cusolverSpHandle_t cusolverH = NULL;
  // GPU does batch QR
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrBandPPT = NULL;
  cusparseMatDescr_t descrBandP = NULL;
  cusolverStatus_t cusolver_status;
  cusparseStatus_t cusparse_status;
  cublasStatus_t cublas_status;

  // create cublas context
  CUBLAS_CALL( cublasCreate(&cublasH) );

  // create cusparse context
  CUSPARSE_CALL( cusparseCreate(&cusparseH) );

  // GPU does batch QR
  // batchsize = lblock
  // m = nblock
  // d_A is CSR format, d_csrValA is of size nnzA*batchSize = 
  // d_x is a matrix of size batchSize * m
  // d_b is a matrix of size batchSize * m
  int *csrRowPtrBandPPT,*csrColIndBandPPT, *csrRowPtrBandP, *csrColIndBandP;
  int *d_csrRowPtrBandPPT,*d_csrColIndBandPPT, *d_csrRowPtrBandP, *d_csrColIndBandP;
  float *csrValBandPPT, *csrValBandP;
  float *d_csrValBandPPT, *d_csrValBandP;
  int ind, u = num_tap - 1;
  size_t size_qr = 0;
  size_t size_internal = 0;
  void *buffer_qr = NULL; // working space for numerical factorization

  // set up banded P matrices
  int mBandP = lblock * ntsblock, nBandP = lblock * nblock;
  int nnzBandP = lblock * num_tap * nblock; 
  csrRowPtrBandP = (int *) malloc((mBandP + 1)*sizeof(int));
  csrColIndBandP = (int *) malloc(nnzBandP*sizeof(int));
  csrValBandP = (float *) malloc(nnzBandP*sizeof(float));

  ind = 0;
  for (int batchId = 0; batchId < lblock; batchId++){
    for (int j = 0; j < ntsblock; j++){
      csrRowPtrBandP[batchId*ntsblock + j] = ind;
      for (int i = 0; i < nblock; i++){
        if (((u+i-j) >= 0) && (i <= j)) {
          csrColIndBandP[ind] = batchId*nblock + i;
          csrValBandP[ind] = hamming(lblock*(j-i) + batchId, num_tap*lblock);
          ++ind;
        }
      }
    }
  }
  csrRowPtrBandP[mBandP] = nnzBandP;

  // set up banded P*P^T matrices
  int mBandPPT = nblock;
  int nnzBandPPT = 2 * (num_tap*nblock - (num_tap - 1) * num_tap / 2) - nblock;
  csrRowPtrBandPPT = (int *) malloc((mBandPPT + 1)*sizeof(int));
  csrColIndBandPPT = (int *) malloc(nnzBandPPT*sizeof(int));
  csrValBandPPT = (float *) malloc(lblock*nnzBandPPT*sizeof(float));

  for (int batchId = 0; batchId < lblock; batchId++){  
    ind = 0;
    // loop over rows
    for (int j=0; j < mBandPPT; j++){
      csrRowPtrBandPPT[j] = ind;
      // loop over columns
      for (int i=0; i < mBandPPT; i++){
        int dji = abs(j-i);
        if ((u-dji) >= 0) {
          float val = 0.;
          for (int k = 0; k < num_tap - dji; k ++) {
            val += hamming(lblock*(k+dji) + batchId, num_tap*lblock) * hamming(lblock*k + batchId, num_tap*lblock);
          }
          csrColIndBandPPT[ind] = i;
          csrValBandPPT[nnzBandPPT*batchId + ind] = val;
          ++ind;
        }
      }
    }
    csrRowPtrBandPPT[mBandPPT] = nnzBandPPT;
  }

  // copy banded P*P^T matrices to device
  CUDA_CALL( cudaMalloc((void **) &d_yh, ntsblock*lblock*sizeof(float)) );
  CUDA_CALL( cudaMalloc((void**)&d_csrValBandPPT, sizeof(float)*nnzBandPPT*lblock) );
  CUDA_CALL( cudaMalloc((void**)&d_csrColIndBandPPT, sizeof(int)*nnzBandPPT) );
  CUDA_CALL( cudaMalloc((void**)&d_csrRowPtrBandPPT, sizeof(int)*(mBandPPT+1)) );
  CUDA_CALL( cudaMalloc((void**)&d_csrValBandP, sizeof(float)*nnzBandP) );
  CUDA_CALL( cudaMalloc((void**)&d_csrColIndBandP, sizeof(int)*nnzBandP) );
  CUDA_CALL( cudaMalloc((void**)&d_csrRowPtrBandP, sizeof(int)*(mBandP+1)) );
  CUDA_CALL( cudaMemcpy(d_csrValBandPPT, csrValBandPPT, sizeof(float)*nnzBandPPT*lblock, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrColIndBandPPT, csrColIndBandPPT, sizeof(int)*nnzBandPPT, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrRowPtrBandPPT, csrRowPtrBandPPT, sizeof(int)*(mBandPPT+1), cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrValBandP, csrValBandP, sizeof(float)*nnzBandP, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrColIndBandP, csrColIndBandP, sizeof(int)*nnzBandP, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrRowPtrBandP, csrRowPtrBandP, sizeof(int)*(mBandP+1), cudaMemcpyHostToDevice) );

  // create cusolver handle, qr info, and matrix descriptor
  CUSOLVER_CALL( cusolverSpCreate(&cusolverH) );
  CUSPARSE_CALL( cusparseCreateMatDescr(&descrBandPPT) );
  CUSPARSE_CALL( cusparseCreateMatDescr(&descrBandP) );
  CUSOLVER_CALL( cusolverSpCreateCsrqrInfo(&info) );	// for batched execution

  // symbolic analysis
  cusolver_status = cusolverSpXcsrqrAnalysisBatched(
		cusolverH, mBandPPT, mBandPPT, nnzBandPPT,
		descrBandPPT, d_csrRowPtrBandPPT, d_csrColIndBandPPT,
		info);
  CUSOLVER_CALL( cusolver_status );

  // prepare working space
  cusolver_status = cusolverSpScsrqrBufferInfoBatched(
		cusolverH, mBandPPT, mBandPPT, nnzBandPPT,
		descrBandPPT, d_csrValBandPPT, d_csrRowPtrBandPPT, d_csrColIndBandPPT,
		lblock,
		info,
		&size_internal,
		&size_qr);
  CUSOLVER_CALL( cusolver_status );

  // generate pseudo timestream
  CUDA_CALL( cudaMalloc((void **) &d_pts, ntsblock*lblock*sizeof(float)) );
  CUFFT_CALL( cufftPlanMany(&plan, 1, &lblock, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, nblock) );
  CUFFT_CALL( cufftExecC2R(plan, d_s, d_yh) );

  // clean up
  free(csrRowPtrBandPPT);
  free(csrColIndBandPPT);
  free(csrValBandPPT);
  CUDA_CALL( cudaFree(buffer_qr) );
  CUDA_CALL( cudaFree(d_yh) );
  CUDA_CALL( cudaFree(d_pts) );
  CUDA_CALL( cudaFree(d_csrValBandPPT) );
  CUDA_CALL( cudaFree(d_csrColIndBandPPT) );
  CUDA_CALL( cudaFree(d_csrRowPtrBandPPT) );
  CUDA_CALL( cudaFree(d_csrValBandP) );
  CUDA_CALL( cudaFree(d_csrColIndBandP) );
  CUDA_CALL( cudaFree(d_csrRowPtrBandP) );
  CUFFT_CALL( cufftDestroy(plan) );
  CUBLAS_CALL( cublasDestroy(cublasH) );
  CUSOLVER_CALL( cusolverSpDestroyCsrqrInfo(info) );
  CUSPARSE_CALL( cusparseDestroyMatDescr(descrBandPPT) );
  CUSPARSE_CALL( cusparseDestroyMatDescr(descrBandP) );
  CUSPARSE_CALL( cusparseDestroy(cusparseH) );
  return 1;
}

/** @brief Transform SWARM spectra into timeseries
 */
int SwarmC2R(aphids_resampler_t *resampler, aphids_context_t *aphids_ctx){
  int i;
  cufftResult cufft_status;
#ifdef GPU_COMPUTE
  //cudaSetDevice(resampler->deviceId);
#ifdef GPU_DEBUG
  float elapsedTime;
  cudaEventRecord(resampler->tic,resampler->stream);
#endif // GPU_DEBUG
  // transform SWARM spectra into timeseries
  for (i = 0; i < resampler->repeat[0]; ++i) {
	cufft_status = cufftExecC2R(resampler->cufft_plan[0],
			resampler->gpu_B_0 + i*resampler->batch[0]*UNPACKED_BENG_CHANNELS, 
			(cufftReal *)resampler->gpu_A_0 + i*resampler->batch[0]*(2*BENG_CHANNELS_));
	cufft_status = cufftExecC2R(resampler->cufft_plan[0],
			resampler->gpu_B_1 + i*resampler->batch[0]*UNPACKED_BENG_CHANNELS, 
			(cufftReal *)resampler->gpu_A_1 + i*resampler->batch[0]*(2*BENG_CHANNELS_));
#ifdef GPU_DEBUG
		if (cufft_status != CUFFT_SUCCESS){
		    hashpipe_error(__FILE__, "CUFFT error: plan 0 execution failed");
		    return STATE_ERROR;
 		}
#endif // GPU_DEBUG
	}
#ifdef GPU_DEBUG
	cudaEventRecord(resampler->toc,resampler->stream);
	cudaEventSynchronize(resampler->toc);
	if (aphids_ctx->iters % APHIDS_UPDATE_EVERY == 0) {
		cudaEventElapsedTime(&elapsedTime,resampler->tic,resampler->toc);
		aphids_log(aphids_ctx, APHIDS_LOG_INFO, "plan 0 cufft took %f ms [%f Gbps]",
			elapsedTime,  2*resampler->repeat[0]*resampler->batch[0]*resampler->fft_size[0] / (elapsedTime * 1e-3) /1e9 /2);
		fprintf(stdout,"GPU_DEBUG : plan 0 took %f ms [%f Gbps] \n", 
			elapsedTime,  2*resampler->repeat[0]*resampler->batch[0]*resampler->fft_size[0] / (elapsedTime * 1e-3) /1e9 /2);
	}
#endif // GPU_DEBUG
#endif // GPU_COMPUTE
  return STATE_PROC;
}

/** @brief Transform SWARM timeseries into R2DBE compatible spectrum
 */
int SwarmR2C(aphids_resampler_t *resampler, aphids_context_t *aphids_ctx){
  int i;
  cufftResult cufft_status;

#ifdef GPU_COMPUTE
  //cudaSetDevice(resampler->deviceId);
#ifdef GPU_DEBUG
  float elapsedTime;
  cudaEventRecord(resampler->tic,resampler->stream);
#endif // GPU_DEBUG
  // transform timeseries into reconfigured spectra
  for (i = 0; i < resampler->repeat[1]; ++i) {
	cufft_status = cufftExecR2C(resampler->cufft_plan[1],
		(cufftReal *) resampler->gpu_A_0 + i*resampler->batch[1]*RESAMPLING_CHUNK_SIZE,
		resampler->gpu_B_0 + i*resampler->batch[1]*(RESAMPLING_CHUNK_SIZE/2+1));
	cufft_status = cufftExecR2C(resampler->cufft_plan[1],
		(cufftReal *) resampler->gpu_A_1 + i*resampler->batch[1]*RESAMPLING_CHUNK_SIZE,
		resampler->gpu_B_1 + i*resampler->batch[1]*(RESAMPLING_CHUNK_SIZE/2+1));
#ifdef GPU_DEBUG
	if (cufft_status != CUFFT_SUCCESS){
	    hashpipe_error(__FILE__, "CUFFT error: plan 1 execution failed");
	    return STATE_ERROR;
 	}
#endif // GPU_DEBUG
  }
#ifdef GPU_DEBUG
  cudaEventRecord(resampler->toc,resampler->stream);
  cudaEventSynchronize(resampler->toc);
  if (aphids_ctx->iters % APHIDS_UPDATE_EVERY == 0) {
	cudaEventElapsedTime(&elapsedTime,resampler->tic,resampler->toc);
	aphids_log(aphids_ctx, APHIDS_LOG_INFO, "plan 1 cufft took %f ms [%f Gbps]",
		elapsedTime,  2*resampler->repeat[1]*resampler->batch[1]*resampler->fft_size[1] / (elapsedTime * 1e-3) /1e9 /2);
	fprintf(stdout,"GPU_DEBUG : plan 1 took %f ms [%f Gbps] \n", 
		elapsedTime,  2*resampler->repeat[1]*resampler->batch[1]*resampler->fft_size[1] / (elapsedTime * 1e-3) /1e9 /2);
	fflush(stdout);
  }
#endif // GPU_DEBUG
#endif // GPU_COMPUTE
  return STATE_PROC;
}

/** @brief Transform half R2DBE spectrum into timeseries
 */
int Hr2dbeC2R(aphids_resampler_t *resampler, aphids_context_t *aphids_ctx){
  int i;
  cufftResult cufft_status;
#ifdef GPU_COMPUTE
  //cudaSetDevice(resampler->deviceId);
#ifdef GPU_DEBUG
  float elapsedTime;
  cudaEventRecord(resampler->tic,resampler->stream);
#endif // GPU_DEBUG
	// mask and transform reconfigured spectra into resampled timeseries 
  for (i = 0; i < resampler->repeat[2]; ++i) {
	cufft_status = cufftExecC2R(resampler->cufft_plan[2],
		resampler->gpu_B_0 + i*resampler->batch[2]*(RESAMPLING_CHUNK_SIZE/2+1) + resampler->skip_chan,
		(cufftReal *) resampler->gpu_A_0 + i*resampler->batch[2]*(RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR));
	cufft_status = cufftExecC2R(resampler->cufft_plan[2],
		resampler->gpu_B_1 + i*resampler->batch[2]*(RESAMPLING_CHUNK_SIZE/2+1) + resampler->skip_chan,
		(cufftReal *) resampler->gpu_A_1 + i*resampler->batch[2]*(RESAMPLING_CHUNK_SIZE*EXPANSION_FACTOR/DECIMATION_FACTOR));
#ifdef GPU_DEBUG
	if (cufft_status != CUFFT_SUCCESS){
	    hashpipe_error(__FILE__, "CUFFT error: plan 2 execution failed");
	    return STATE_ERROR;
 	}
#endif // GPU_DEBUG
  }
#ifdef GPU_DEBUG
  cudaEventRecord(resampler->toc,resampler->stream);
  cudaEventSynchronize(resampler->toc);
  if (aphids_ctx->iters % APHIDS_UPDATE_EVERY == 0) {
	cudaEventElapsedTime(&elapsedTime,resampler->tic,resampler->toc);
	aphids_log(aphids_ctx, APHIDS_LOG_INFO, "plan 2 cufft took %f ms [%f Gbps]",
		elapsedTime,  2*resampler->repeat[2]*resampler->batch[2]*resampler->fft_size[2] / (elapsedTime * 1e-3) /1e9 /2);
	fprintf(stdout,"GPU_DEBUG : plan 2 took %f ms [%f Gbps] \n", 
		elapsedTime,  2*resampler->repeat[2]*resampler->batch[2]*resampler->fft_size[2] / (elapsedTime * 1e-3) /1e9 /2);
	fflush(stdout);
  }
#endif // GPU_DEBUG
#endif // GPU_COMPUTE
  return STATE_PROC;
}

// destructor
int aphids_resampler_destroy(aphids_resampler_t *resampler) {
  int i;
  cufftResult cufft_status;
  HANDLE_ERROR( cudaSetDevice(resampler->deviceId) );
  HANDLE_ERROR( cudaFree(resampler->gpu_A_0) );
  HANDLE_ERROR( cudaFree(resampler->gpu_B_0) );
  HANDLE_ERROR( cudaFree(resampler->gpu_A_1) );
  HANDLE_ERROR( cudaFree(resampler->gpu_B_1) );
  HANDLE_ERROR( cudaFree(resampler->gpu_out_buf) );
  for (i=0; i < 3; ++i){
	cufft_status = cufftDestroy(resampler->cufft_plan[i]);
	if (cufft_status != CUFFT_SUCCESS){
#ifdef GPU_DEBUG
          hashpipe_error(__FILE__, "CUFFT error: problem destroying plan %d\n", i);
#endif
	} 
  }
#ifdef GPU_DEBUG
  cudaEventDestroy(resampler->tic);
  cudaEventDestroy(resampler->toc);
  HANDLE_ERROR( cudaStreamDestroy(resampler->stream) );
#endif
  HANDLE_ERROR( cudaDeviceReset() );
  return 1;
}

static void *run_method(hashpipe_thread_args_t * args) {

  int i = 0;
  int rv = 0;
  int index_in = 0;
  int index_out = 0;
  beng_group_completion_t this_bgc;
  vdif_in_databuf_t *db_in = (vdif_in_databuf_t *)args->ibuf;
  vdif_out_databuf_t *db_out = (vdif_out_databuf_t *)args->obuf;
  aphids_context_t aphids_ctx;
  int state = STATE_INIT;

  aphids_resampler_t *resampler;
  dim3 threads, blocks;

  // initialize the aphids context
  rv = aphids_init(&aphids_ctx, args);
  if (rv != APHIDS_OK) {
    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
    return NULL;
  }


  /* Initialize GPU  */
  //~ fprintf(stdout, "sizeof(vdif_in_packet_block_t): %d\n", sizeof(vdif_in_packet_block_t));
  //~ fprintf(stdout, "sizeof(vdif_out_packet_block_t): %d\n", sizeof(vdif_out_packet_block_t));

  // initalize resampler
  resampler = (aphids_resampler_t *) malloc(NUM_GPU*sizeof(aphids_resampler_t));
  for (i=0; i< NUM_GPU; i++){
    aphids_resampler_init(&(resampler[i]), i);
  }


  while (run_threads()) { // hashpipe wants us to keep running

    switch(state) {

    case STATE_ERROR:

      {

	// set status to show we're in error
	aphids_set(&aphids_ctx, "status", "error");

	// do nothing, wait to be killed
	sleep(1);

	break;

      }

    case STATE_INIT:

      {

	// set status to show we're going to run GPU code
	aphids_set(&aphids_ctx, "status", "processing data on GPU");

	// and set our next state
	state = STATE_PROC;

      }

    case STATE_PROC:

      {

	// check if next block filled in input buffer
	while ((rv = hashpipe_databuf_wait_filled((hashpipe_databuf_t *)db_in, index_in)) != HASHPIPE_OK) {

	  if (rv == HASHPIPE_TIMEOUT) { // index_in is not ready
	    aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe input databuf timeout");

	    // need to check run_threads here again
	    if (run_threads())
	      continue;
	    else
	      break;

	  } else { // any other return value is an error

	    // raise an error and exit thread
	    hashpipe_error(__FUNCTION__, "error waiting for filled databuf in %s:%s(%d)",__FILE__,__FUNCTION__,__LINE__);
	    state = STATE_ERROR;
	    break;

	  }

	}
	
	aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "hashpipe databuf filled");
	
	fprintf(stdout,"%s:%s(%d): input buffer %d filled\n",__FILE__,__FUNCTION__,__LINE__,index_in);

	// check if next block free in output buffer; this could possibly
	// be moved down to just before/after last link in GPU process chain,
	// but for now since memory in GPU is being reused by multiple 
	// links, wait for output to be free before doing anything
	while ((rv = hashpipe_databuf_wait_free((hashpipe_databuf_t *)db_out, index_out)) != HASHPIPE_OK) {

	  if (rv == HASHPIPE_TIMEOUT) { // index_out is not ready
	    aphids_log(&aphids_ctx, APHIDS_LOG_ERROR, "hashpipe output databuf timeout");
	    continue;

	  } else { // any other return value is an error

	    // raise an error and exit thread
	    hashpipe_error(__FILE__, "error waiting for free databuf");
	    state = STATE_ERROR;
	    break;

	  }

	}
	
	aphids_log(&aphids_ctx, APHIDS_LOG_INFO, "hashpipe databuf free");
	
	fprintf(stdout,"%s:%s(%d): output buffer %d free\n",__FILE__,__FUNCTION__,__LINE__,index_out);
	
	// grab the block at index_in from input buffer
	this_bgc = (beng_group_completion_t)db_in->bgc[index_in];
	
	// Set GPU, ID is embedded in input buffer block. The vdif-in-thread
	// will have copied the data to a particular GPU, so here need to 
	// select the same one when processing the next dataset.
	i = this_bgc.gpu_id;
	cudaSetDevice(i);
	resampler[i].deviceId = this_bgc.gpu_id;
	
	// If any correction needed for missing data, probably need to do
	// it here; e.g. the received data tells exactly which parts of the
	// spectrum belonging to which B-engine counter values are missing,
	// if any, so an easy remedy could be to have a set of random B-eng-
	// over-VDIF data payloads ready, and then ship the required number
	// packets with the correct header information inserted to GPU 
	// memory to get a complete group of B-engine frames. The GPU memory
	// buffer is exactly the required size for BENG_FRAMES_PER_GROUP 
	// number of B-engine frames' worth of VDIF packets.
	
	uint64_t b_zero = this_bgc.bfc[0].b;
	// de-packetize vdif
	// call to vdif_to_beng?
        threads.x = 32; threads.y = 32; threads.z = 1;
        blocks.x = 128, blocks.y = 1; blocks.z = 1; 
        vdif_to_beng<<<blocks,threads>>>((int32_t*) this_bgc.bgv_buf_gpu, 
					resampler[i].gpu_A_0, resampler[i].gpu_A_1,
					this_bgc.beng_group_vdif_packet_count,b_zero);
	
	// ... set the input buffer block free ...
	hashpipe_databuf_set_free((hashpipe_databuf_t *)db_in, index_in);
	// ... and increment input buffer index.
	fprintf(stdout,"%s:%s(%d): input buffer %d free\n",__FILE__,__FUNCTION__,__LINE__,index_in);
	index_in = (index_in + 1) % db_in->header.n_block;

	
	// In principle we can start processing on next input block on a 
	// separate GPU, for now serialize GPU work, but use different ones
	// in turn. To work all GPUs in parallel, need the following?:
	//   * keep set of processing state indicators, one for each GPU
	//   * on each pass, depending on GPU, check if task n is complete, 
	//     and if so, start task n+1 and update state indicator
	//   * needs moving wait-filled/free around and dependent on the
	//     processing state for each GPU

	// reorder BENG data
	//~ already set device: cudaSetDevice(i);
        threads.x = 16; threads.y = 16; threads.z = 1;
        blocks.x = (BENG_CHANNELS_*BENG_SNAPSHOTS/(16*16)); blocks.y = 1; blocks.z = 1;
	reorderTzp_smem<<<blocks,threads>>>(resampler[i].gpu_A_0, resampler[i].gpu_B_0, BENG_BUFFER_IN_COUNTS);
	reorderTzp_smem<<<blocks,threads>>>(resampler[i].gpu_A_1, resampler[i].gpu_B_1, BENG_BUFFER_IN_COUNTS);

	// transform SWARM spectra to time series
	state = SwarmC2R(&(resampler[i]), &aphids_ctx);

	// transform SWARM time series to R2DBE compatible spectra
	state = SwarmR2C(&(resampler[i]), &aphids_ctx);

	// transform R2DBE spectra to trimmed and resampled time series
	state = Hr2dbeC2R(&(resampler[i]), &aphids_ctx);

	// calculate threshold for quantization
#ifdef QUANTIZE_THRESHOLD_COMPUTE
	if (resampler[i].quantizeThreshold_0 == 0){

          // wrap as device pointer
          thrust::device_ptr<float> dev_ptr;

          // setup arguments
          summary_stats_unary_op<float>  unary_op;
          summary_stats_binary_op<float> binary_op;
          summary_stats_data<float>      init,result;

	  // calculate variance for phased sums independently
  	  init.initialize();

	  dev_ptr = thrust::device_pointer_cast((float *) resampler[i].gpu_A_0);
          result = thrust::transform_reduce(dev_ptr, 
						dev_ptr + 2*BENG_CHANNELS_*BENG_SNAPSHOTS,
						unary_op, init, binary_op);
	  resampler[i].quantizeThreshold_0 = sqrt(result.variance());
	  // add computation of mean
	  resampler[i].quantizeOffset_0 = result.average();

	  dev_ptr = thrust::device_pointer_cast((float *) resampler[i].gpu_A_1);
  	  init.initialize();
          result = thrust::transform_reduce(dev_ptr, 
						dev_ptr + 2*BENG_CHANNELS_*BENG_SNAPSHOTS,
						unary_op, init, binary_op);
	  resampler[i].quantizeThreshold_1 = sqrt(result.variance());
	  // add computation of mean
	  resampler[i].quantizeOffset_1 = result.average();

      // set the same quantization parameters across all resamplers
      for (int ii=0; ii < NUM_GPU; ii++) {
		  resampler[ii].quantizeThreshold_0 = resampler[i].quantizeThreshold_0;
		  resampler[ii].quantizeOffset_0 = resampler[i].quantizeOffset_0;
		  resampler[ii].quantizeThreshold_1 = resampler[i].quantizeThreshold_1;
		  resampler[ii].quantizeOffset_1 = resampler[i].quantizeOffset_1;
	  }
	  fprintf(stdout,"%s:%s(%d): Quantization parameters set to {0:(%f,%f),1:(%f,%f)}\n",__FILE__,__FUNCTION__,__LINE__,resampler[i].quantizeOffset_0,resampler[i].quantizeThreshold_0,resampler[i].quantizeOffset_1,resampler[i].quantizeThreshold_1);
    }
#endif

	// quantize to 2-bits
	//~ already set device: cudaSetDevice(i);
	threads.x = 16; threads.y = 32; threads.z = 1;
	blocks.x = 512; blocks.y = 1; blocks.z = 1;
	quantize2bit<<<blocks,threads>>>((float *) resampler[i].gpu_A_0, (unsigned int*) resampler[i].gpu_B_0, 
	(2*BENG_CHANNELS_*BENG_SNAPSHOTS*EXPANSION_FACTOR),
	resampler[i].quantizeThreshold_0,resampler[i].quantizeOffset_0);
	quantize2bit<<<blocks,threads>>>((float *) resampler[i].gpu_A_1, (unsigned int*) resampler[i].gpu_B_1, 
	(2*BENG_CHANNELS_*BENG_SNAPSHOTS*EXPANSION_FACTOR),
	resampler[i].quantizeThreshold_1,resampler[i].quantizeOffset_1);
	
	// copy data to output buffer
	cudaMemcpy((void *)resampler[i].gpu_out_buf,(void *)resampler[i].gpu_B_0,sizeof(vdif_out_data_block_t),cudaMemcpyDeviceToDevice);
	cudaMemcpy((void *)resampler[i].gpu_out_buf + sizeof(vdif_out_data_block_t),(void *)resampler[i].gpu_B_1,sizeof(vdif_out_data_block_t),cudaMemcpyDeviceToDevice);

	// Output to next thread to mirror the input?:
	//   * update metadata that describes the amount of data available
	db_out->blocks[index_out].bit_depth = 2;
	db_out->blocks[index_out].N_32bit_words_per_chan = (2*BENG_CHANNELS_*BENG_SNAPSHOTS*EXPANSION_FACTOR) / (32 / db_out->blocks[index_out].bit_depth);
	db_out->blocks[index_out].gpu_id = resampler[i].deviceId; //index_out % NUM_GPU;
	db_out->blocks[index_out].vdg_buf_gpu = resampler[i].gpu_out_buf;
	// copy the template VDIF header over to output
	memcpy(&db_out->blocks[index_out].vdif_header_template,&this_bgc.vdif_header_template,sizeof(vdif_in_header_t));
	
	// let hashpipe know we're done with the buffer (for now) ...
	hashpipe_databuf_set_filled((hashpipe_databuf_t *)db_out, index_out);
	// .. and update the index modulo the maximum buffer depth
	fprintf(stdout,"%s:%s(%d): output buffer %d filled\n",__FILE__,__FUNCTION__,__LINE__,index_out);
	index_out = (index_out + 1) % db_out->header.n_block;
	

	// update aphids statistics
	aphids_update(&aphids_ctx);

	break;
      } // case STATE_PROC

    } // switch(state)

  } // end while(run_threads())

  /* GPU clean-up code */
  for (i=0; i<NUM_GPU; i++)
    aphids_resampler_destroy(&(resampler[i]));
  free(resampler);

  // destroy aphids context and exit
  aphids_destroy(&aphids_ctx);

  return NULL;
}

static hashpipe_thread_desc_t vdif_inout_gpu_thread = {
 name: "vdif_inout_gpu_thread",
 skey: "VDIFIO",
 init: NULL,
 run:  run_method,
 ibuf_desc: {vdif_in_databuf_create},
 obuf_desc: {vdif_out_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&vdif_inout_gpu_thread);
}
