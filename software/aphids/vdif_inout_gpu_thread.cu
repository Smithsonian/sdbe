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
    // Misc
    // GPU device ID associated with this resampler
    int deviceId;
    // number of channels to skip at start of SwarmC2R (2nd stage)
    int skip_chan;
    // Requantization
    float quantizeThreshold_0, quantizeThreshold_1;
    float quantizeOffset_0, quantizeOffset_1;
    summary_stats_data<float> ssd;
    // Data buffers
    // hold all half-fluffed B-engine data
    int8_t *beng_0, *beng_1;
    // hold batch fully-fluffed B-engine data
    cufftComplex *beng_fluffed_0, *beng_fluffed_1;
    // hold batch 1st stage IFFT output
    cufftReal *swarm_c2r_ifft_out_0, *swarm_c2r_ifft_out_1;
    // hold batch 2nd stage FFT output
    cufftComplex *swarm_r2c_fft_out_0, *swarm_r2c_fft_out_1;
    // hold batch 3rd stage IFFT output
    cufftReal *r2dbe_c2r_ifft_out_0, *r2dbe_c2r_ifft_out_1;
    // hold all requantized output
    vdif_out_data_group_t *gpu_out_buf;
    // FFT parameters
    // sizes for stages 1, 2, 3
    int fft_size[3];
    // batches for stages 1, 2, 3
    int batch[3];
    // CUFFT plans for stages 1, 2, 3
    cufftHandle cufft_plan[3];
#ifdef GPU_DEBUG
    // Debugging
    cudaStream_t stream;
    cudaEvent_t tic,toc;
#endif

} aphids_resampler_t;

/** @brief Initialize resampler structure (including device memory).
 */
int aphids_resampler_init(aphids_resampler_t *resampler, int skip_mhz, int _deviceId) {
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
  
  ///////////////// Allocate device memory /////////////////////////////
  /* beng_[01] each stores a single 2-bit sample in each byte. Total
   * size requirement is:
   *     16384 -- number of channels [BENG_CHANNELS_]
   *   x   128 -- number of snapshots per frame [BENG_SNAPSHOTS]
   *   x   143 -- number of frames [BENG_FRAMES_PER_GROUP]
   *   x     2 -- one real, one imaginary
   *   = 599785472 bytes per polarization
   */
  cudaMalloc((void **)&(resampler->beng_0), BENG_FRAMES_PER_GROUP*BENG_SNAPSHOTS*BENG_CHANNELS_*2*sizeof(int8_t));
  cudaMalloc((void **)&(resampler->beng_1), BENG_FRAMES_PER_GROUP*BENG_SNAPSHOTS*BENG_CHANNELS_*2*sizeof(int8_t));
  /* beng_fluffed_[01] will hold RESAMPLE_BATCH_SNAPSHOTS number of
   * B-engine snapshots as full cufftComplex values. The total size
   * requirement is:
   *     16385 -- number of channels in C2R [BENG_CHANNELS]
   *   x   143 -- snapshots per batch [RESAMPLE_BATCH_SNAPSHOTS]
   *   x     8 -- sizeof(cufftComplex)
   *   = 18744440 bytes per polarization
   */
  cudaMalloc((void **)&(resampler->beng_fluffed_0), RESAMPLE_BATCH_SNAPSHOTS*BENG_CHANNELS*sizeof(cufftComplex));
  cudaMalloc((void **)&(resampler->beng_fluffed_1), RESAMPLE_BATCH_SNAPSHOTS*BENG_CHANNELS*sizeof(cufftComplex));
  /* swarm_c2r_ifft_out_[01] will hold the output of the SwarmC2R
   * inverse Fourier transform. The total size requirement is:
   *     32768 -- number of real-valued samples [2*BENG_CHANNELS_]
   *   x   143 -- snapshots per batch [FFT_BATCHES_SWARM_C2R]
   *   x     4 -- sizeof(cufftReal)
   *   = 18743296 bytes per polarization
   */
  cudaMalloc((void **)&(resampler->swarm_c2r_ifft_out_0), FFT_BATCHES_SWARM_C2R*2*BENG_CHANNELS_*sizeof(cufftReal));
  cudaMalloc((void **)&(resampler->swarm_c2r_ifft_out_1), FFT_BATCHES_SWARM_C2R*2*BENG_CHANNELS_*sizeof(cufftReal));
  /* swarm_r2c_fft_out_[01] will hold the output of the SwarmR2C
   * Fourier transform. The total size requirement is:
   *     18305 -- number of channels output [FFT_SIZE_SWARM_R2C/2 + 1]
   *   x   128 -- number FFT windows in batch [FFT_BATCHES_SWARM_R2C]
   *   x     8 -- sizeof(cufftComplex)
   *   = 18744320 bytes per polarization
   */
  cudaMalloc((void **)&(resampler->swarm_r2c_fft_out_0), FFT_BATCHES_SWARM_R2C*(FFT_SIZE_SWARM_R2C/2 + 1)*sizeof(cufftComplex));
  cudaMalloc((void **)&(resampler->swarm_r2c_fft_out_1), FFT_BATCHES_SWARM_R2C*(FFT_SIZE_SWARM_R2C/2 + 1)*sizeof(cufftComplex));
  /* r2dbe_c2r_ifft_out_[01] will hold the output of the R2dbeC2R
   * inverse Fourier transform. The total size requirement is:
   *     32768 -- number of real-valued samples [FFT_SIZE_R2DBE_C2R]
   *   x   128 -- number of FFT windows in batch [FFT_BATCHES_R2DBE_C2R]
   *   x     4 -- sizeof(cufftReal)
   *   = 16777216 bytes per polarization
   */
  cudaMalloc((void **)&(resampler->r2dbe_c2r_ifft_out_0), FFT_BATCHES_R2DBE_C2R*FFT_SIZE_R2DBE_C2R*sizeof(cufftReal));
  cudaMalloc((void **)&(resampler->r2dbe_c2r_ifft_out_1), FFT_BATCHES_R2DBE_C2R*FFT_SIZE_R2DBE_C2R*sizeof(cufftReal));
  /* Holds the requantized 2-bit output data for entire group of
   * B-frames processed. Total size requirement is:
   *     sizeof(vdif_out_group_t) -- this is everything
   *   =     2 -- number of polarizations [VDIF_CHAN]
   *   x 16384 -- number of VDIF packets per group [VDIF_OUT_PKTS_PER_BLOCK]
   *   x  8192 -- number of bytes per VDIF packet data [VDIF_OUT_PKT_DATA_SIZE]
   *   = 268435456 bytes total (single structure for both polarizations)
   */
  cudaMalloc((void **)&(resampler->gpu_out_buf), sizeof(vdif_out_data_group_t));
  //////// Total allocated memory in this batch: 1614024944 bytes //////
  
  /*
   * The number of skipped channels is needed before setting parameters 
   * of the third transform, so define it at the start.
   */
  resampler->skip_chan = (int)(skip_mhz * 1e6 / (SWARM_RATE / FFT_SIZE_SWARM_R2C));
  
  // 1st stage: SwarmC2R IFFT
  resampler->fft_size[0] = FFT_SIZE_SWARM_C2R;
  inembed[0] = FFT_SIZE_SWARM_C2R/2 + 1; 
  onembed[0] = FFT_SIZE_SWARM_C2R;
  resampler->batch[0] = FFT_BATCHES_SWARM_C2R;
  cufft_status = cufftPlanMany(
    &(resampler->cufft_plan[0]), // plan handle
    1,                           // rank
    &(resampler->fft_size[0]),   // FFT size
    inembed,1,inembed[0],        // input: dims,dim0 stride,batch stride
    onembed,1,onembed[0],        // output: dims,dim0 stride,batch stride
    CUFFT_C2R,                   // type of transform
    resampler->batch[0]);        // batches
  if (cufft_status != CUFFT_SUCCESS) {
    hashpipe_error(__FILE__, "CUFFT error: plan 0 creation failed");
  }
#ifdef GPU_DEBUG
  cufftGetSize(resampler->cufft_plan[0],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 0 is %dx%d\n", resampler->batch[0],resampler->fft_size[0]);
  fprintf(stdout,"GPU_DEBUG : plan 0 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  
  
  // 2nd stage: SwarmR2C FFT
  resampler->fft_size[1] = FFT_SIZE_SWARM_R2C;
  inembed[0] = FFT_SIZE_SWARM_R2C; 
  onembed[0] = FFT_SIZE_SWARM_R2C/2 + 1;
  resampler->batch[1] = FFT_BATCHES_SWARM_R2C;
  cufft_status = cufftPlanMany(
    &(resampler->cufft_plan[1]), // plan handle
    1,                           // rank
    &(resampler->fft_size[1]),   // FFT size
    inembed,1,inembed[0],        // input: dims,dim0 stride,batch stride
    onembed,1,onembed[0],        // output: dims,dim0 stride,batch stride
    CUFFT_R2C,                   // type of transform
    resampler->batch[1]);        // batches
  if (cufft_status != CUFFT_SUCCESS) {
    hashpipe_error(__FILE__, "CUFFT error: plan 1 creation failed");
  }
#ifdef GPU_DEBUG
  cufftGetSize(resampler->cufft_plan[1],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 1 is %dx%d\n", resampler->batch[1],resampler->fft_size[1]);
  fprintf(stdout,"GPU_DEBUG : plan 1 worksize: %u\n",(unsigned) workSize[0]);
  reportDeviceMemInfo();
#endif  
  
  // 3rd stage: R2dbeC2R IFFT
  resampler->fft_size[2] = FFT_SIZE_R2DBE_C2R;
  inembed[0] = FFT_SIZE_R2DBE_C2R/2 + 1;
  onembed[0] = FFT_SIZE_R2DBE_C2R;
  resampler->batch[2] = FFT_BATCHES_R2DBE_C2R;
  cufft_status = cufftPlanMany(
    &(resampler->cufft_plan[2]), // plan handle
    1,                           // rank
    &(resampler->fft_size[2]),   // FFT size
    inembed,1,                   // input: dims,dim0 stride
    FFT_SIZE_SWARM_R2C/2 + 1,    // input batch stride, NOTE: this matches output batch stride in 2nd stage
    onembed,1,onembed[0],        // output: dims,dim0 stride,batch stride
    CUFFT_C2R,                   // type of transform
    resampler->batch[2]);        // batches
  if (cufft_status != CUFFT_SUCCESS) {
    hashpipe_error(__FILE__, "CUFFT error: plan 2 creation failed");
  }
  
#ifdef GPU_DEBUG
  for (i=0; i<3; ++i){
    cufftSetStream(resampler->cufft_plan[i], resampler->stream);
  }
  cufftGetSize(resampler->cufft_plan[2],workSize);
  fprintf(stdout,"GPU_DEBUG : plan 2 is %dx%d\n", resampler->batch[2],resampler->fft_size[2]);
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

/*
 * Read 2-bit value as int8_t and shift input data accordingly inplace.
 */
__device__ int8_t read_2bit_sample(int32_t *samples_int) {
  int8_t sample;
  sample = (*samples_int & 0x03) - 2;
  *samples_int = (*samples_int) >> 2;
  return sample;
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
 int8_t *beng_data_out_0,
 int8_t *beng_data_out_1,
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
    /* Reorder to have consecutive channels for a single FFT window 
     * adjacent in memory. No transpose is therefore needed to have the 
     * IFFT operate on contiguous memory.
     */
    idx_beng_data_out = ((cid>>1)*(2*SWARM_XENG_PARALLEL_CHAN) + (cid%2) + fid*2)*SWARM_N_FIDS;
    idx_beng_data_out += (bcount % BENG_FRAMES_PER_GROUP)*BENG_SNAPSHOTS*BENG_CHANNELS_;
    /* Output buffers reference int8_t data, and data for each channel 
     * will occupy two fields, re+im.
     */
    idx_beng_data_out *= 2;
    /* idata increases by the number of int32_t handled simultaneously
     * by all x-threads. Each thread handles B-engine packet data 
     * for a single snapshot per iteration.
     */
    for (idata=0; idata<VDIF_INT_SIZE_DATA; idata+=BENG_VDIF_INT_PER_SNAPSHOT*blockDim.x){
      int this_snapshot;
      // Calculate the snapshot number within the B-frame...
      this_snapshot = idata/BENG_VDIF_INT_PER_SNAPSHOT + threadIdx.x;
      /* Get sample data out of global memory. Offset from the 
       * VDIF frame start by the header, the number of snapshots
       * processed by the group of x-threads (idata), and the
       * particular snapshot offset for THIS x-thread 
       * (BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x).
       */
      samples_per_snapshot_half_0 = *(vdif_frame_start + VDIF_INT_SIZE_HEADER + idata + BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x);
      samples_per_snapshot_half_1 = *(vdif_frame_start + VDIF_INT_SIZE_HEADER + idata + BENG_VDIF_INT_PER_SNAPSHOT*threadIdx.x + 1);
      /* 2-bit samples are stored in order:
       *   byte0: d1_im d1_re d0_im d0_re
       *   byte1: c1_im c1_re c0_im c0_re
       *   byte2: b1_im b1_re b0_im b0_re
       *   byte3: a1_im a1_re a0_im a0_re
       *   byte4: h1_im h1_re h0_im h0_re
       *   byte5: g1_im g1_re g0_im g0_re
       *   byte6: f1_im f1_re f0_im f0_re
       *   byte7: e1_im e1_re e0_im e0_re
       * and each group of SWARM_XENG_PARALLEL_CHAN is [a b c d e f g h]
       */
      for (isample=0; isample<SWARM_XENG_PARALLEL_CHAN/2; ++isample){
        int this_idx;
        //          (fid,cid,bcount)                       ( ordering {d..a})
        this_idx = idx_beng_data_out + 2*(SWARM_XENG_PARALLEL_CHAN/2-(isample+1));
        // Adjust the index according to the snapshot number.
        this_idx += this_snapshot*2*BENG_CHANNELS_;
        //    (pol X/Y)      (im/re)
        beng_data_out_1[this_idx + 0] = read_2bit_sample(&samples_per_snapshot_half_0); // imaginary
        beng_data_out_1[this_idx + 1] = read_2bit_sample(&samples_per_snapshot_half_0); // real
        beng_data_out_0[this_idx + 0] = read_2bit_sample(&samples_per_snapshot_half_0); // imaginary
        beng_data_out_0[this_idx + 1] = read_2bit_sample(&samples_per_snapshot_half_0); // real
        //          (fid,cid,bcount)                     ( ordering {h..e})
        this_idx = idx_beng_data_out + 2*(SWARM_XENG_PARALLEL_CHAN-(isample+1));
        // Adjust the index according to the snapshot number.
        this_idx += this_snapshot*2*BENG_CHANNELS_;
        //    (pol X/Y)       (im/re)
        beng_data_out_1[this_idx + 0] = read_2bit_sample(&samples_per_snapshot_half_1); // imaginary
        beng_data_out_1[this_idx + 1] = read_2bit_sample(&samples_per_snapshot_half_1); // real
        beng_data_out_0[this_idx + 0] = read_2bit_sample(&samples_per_snapshot_half_1); // imaginary
        beng_data_out_0[this_idx + 1] = read_2bit_sample(&samples_per_snapshot_half_1); // real
      }
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

/* Convert int8_t,int8_t spectral data to cufftComplex spectral data.
 * Arguments:
 * ----------
 *   in_spec_start -- pointer to first sample of first spectrum in input
 *   out_spec_start -- pointer to first sample of first spectrum in output
 *   nspectra -- number of spectra in input to process
 * Returns:
 * --------
 *   <void>
 * Notes:
 * ------
 *   The input spectra are assumed to be written as imag,real,imag,real...
 *   with each imag,real pair representing a single complex-valued spectral
 *   sample written as two consecutive int8_t values.
 *
 *   An additional zero-valued channel is inserted in the output for each
 *   spectrum at the half-sample-rate position as is required for the subsequent
 *   C2R transform.
 *
 *   Call with 32 x-threads, 32 y-threads, 16 x-blocks = 16384 channels
 *   = 1 spectrum per pass. So the loop is only over the number of spectra
 *   to be processed.
 */
__global__ void beng_int8_to_cufftComplex(const int8_t * __restrict__ in_spec_start, cufftComplex *out_spec_start, int nspectra)
{
	// write single channel cufftComplex pair per thread
	int idx_out = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*blockDim.x*blockDim.y;
	// read single channel int8_t,int8_t pair per thread
	int idx_in = 2*(idx_out);
	int ii;
	float im_in, re_in;
	cufftComplex zz_out;
	for (ii=0; ii<nspectra; ii++) {
		// read imag,real and write cufftComplex
		im_in = __int2float_rd(*(in_spec_start + idx_in));
		re_in = __int2float_rd(*(in_spec_start + idx_in + 1));
		zz_out = make_cuFloatComplex(re_in, im_in);
		*(out_spec_start+idx_out) = zz_out;
		/* Advance input pointer by a single spectrum, BENG_CHANNELS_ number
		 * of channels at 2 x sizeof(int8_t) each
		 */
		idx_in += 2*BENG_CHANNELS_;
		/* Advance output poniter by a single spectrum, BENG_CHANNELS number
		 * of channels at 1 x sizeof(cufftComplex) each
		 */
		idx_out += BENG_CHANNELS;
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

/* 1st stage: SwarmC2R IFFT
 * input: beng_fluffed_[01]
 * output: swarm_c2r_ifft_out_[01]
 */
int SwarmC2R(aphids_resampler_t *resampler, aphids_context_t *aphids_ctx) {
  cufftResult cufft_status;
  
#ifdef GPU_COMPUTE
#ifdef GPU_DEBUG
  float elapsedTime;
  cudaEventRecord(resampler->tic,resampler->stream);
#endif // GPU_DEBUG
  // do transform for pol 0
  cufft_status = cufftExecC2R(resampler->cufft_plan[0],
    resampler->beng_fluffed_0, resampler->swarm_c2r_ifft_out_0);
  if (cufft_status != CUFFT_SUCCESS){
    hashpipe_error(__FILE__, "CUFFT error: plan 0 execution failed (pol 0)");
    return STATE_ERROR;
  }
  // do transform for pol 1
  cufft_status = cufftExecC2R(resampler->cufft_plan[0],
    resampler->beng_fluffed_1, resampler->swarm_c2r_ifft_out_1);
  if (cufft_status != CUFFT_SUCCESS){
    hashpipe_error(__FILE__, "CUFFT error: plan 0 execution failed (pol 1)");
    return STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cudaEventRecord(resampler->toc,resampler->stream);
  cudaEventSynchronize(resampler->toc);
  if (aphids_ctx->iters % APHIDS_UPDATE_EVERY == 0) {
    cudaEventElapsedTime(&elapsedTime,resampler->tic,resampler->toc);
    aphids_log(aphids_ctx, APHIDS_LOG_INFO, "plan 0 cufft took %f ms [%f Gbps]",
      elapsedTime, resampler->batch[0]*resampler->fft_size[0] / (elapsedTime * 1e-3) /1e9 /2);
    fprintf(stdout,"GPU_DEBUG : plan 0 took %f ms [%f Gbps] \n", 
      elapsedTime, resampler->batch[0]*resampler->fft_size[0] / (elapsedTime * 1e-3) /1e9 /2);
    fflush(stdout);
  }
#endif // GPU_DEBUG
#endif // GPU_COMPUTE
  return STATE_PROC;
}

/* 2nd stage: SwarmR2C FFT
 * input: swarm_c2r_ifft_out_[01]
 * output: swarm_r2c_fft_out_[01]
 */
int SwarmR2C(aphids_resampler_t *resampler, aphids_context_t *aphids_ctx) {
  cufftResult cufft_status;
  
#ifdef GPU_COMPUTE
#ifdef GPU_DEBUG
  float elapsedTime;
  cudaEventRecord(resampler->tic,resampler->stream);
#endif // GPU_DEBUG
  // do transform for pol 0
  cufft_status = cufftExecR2C(resampler->cufft_plan[1],
    resampler->swarm_c2r_ifft_out_0, resampler->swarm_r2c_fft_out_0);
  if (cufft_status != CUFFT_SUCCESS){
    hashpipe_error(__FILE__, "CUFFT error: plan 1 execution failed (pol 0)");
    return STATE_ERROR;
  }
  // do transform for pol 1
  cufft_status = cufftExecR2C(resampler->cufft_plan[1],
    resampler->swarm_c2r_ifft_out_1, resampler->swarm_r2c_fft_out_1);
  if (cufft_status != CUFFT_SUCCESS){
    hashpipe_error(__FILE__, "CUFFT error: plan 1 execution failed (pol 1)");
    return STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cudaEventRecord(resampler->toc,resampler->stream);
  cudaEventSynchronize(resampler->toc);
  if (aphids_ctx->iters % APHIDS_UPDATE_EVERY == 0) {
    cudaEventElapsedTime(&elapsedTime,resampler->tic,resampler->toc);
    aphids_log(aphids_ctx, APHIDS_LOG_INFO, "plan 1 cufft took %f ms [%f Gbps]",
      elapsedTime, resampler->batch[1]*resampler->fft_size[1] / (elapsedTime * 1e-3) /1e9 /2);
    fprintf(stdout,"GPU_DEBUG : plan 1 took %f ms [%f Gbps] \n", 
      elapsedTime, resampler->batch[1]*resampler->fft_size[1] / (elapsedTime * 1e-3) /1e9 /2);
    fflush(stdout);
  }
#endif // GPU_DEBUG
#endif // GPU_COMPUTE
  return STATE_PROC;
}

/* 3rd stage: R2dbeC2R IFFT
 * input: swarm_r2c_fft_out_[01]
 * output: r2dbe_c2r_ifft_out_[01]
 */
int R2dbeC2R(aphids_resampler_t *resampler, aphids_context_t *aphids_ctx) {
  cufftResult cufft_status;
  
#ifdef GPU_COMPUTE
#ifdef GPU_DEBUG
  float elapsedTime;
  cudaEventRecord(resampler->tic,resampler->stream);
#endif // GPU_DEBUG
  // do transform for pol 0, skipping integer MHz at the start
  cufft_status = cufftExecC2R(resampler->cufft_plan[2],
    resampler->swarm_r2c_fft_out_0+resampler->skip_chan, resampler->r2dbe_c2r_ifft_out_0);
  if (cufft_status != CUFFT_SUCCESS){
    hashpipe_error(__FILE__, "CUFFT error: plan 2 execution failed (pol 0)");
    return STATE_ERROR;
  }
  // do transform for pol 1, skipping integer MHz at the start
  cufft_status = cufftExecC2R(resampler->cufft_plan[2],
    resampler->swarm_r2c_fft_out_1+resampler->skip_chan, resampler->r2dbe_c2r_ifft_out_1);
  if (cufft_status != CUFFT_SUCCESS){
    hashpipe_error(__FILE__, "CUFFT error: plan 2 execution failed (pol 1)");
    return STATE_ERROR;
  }
#ifdef GPU_DEBUG
  cudaEventRecord(resampler->toc,resampler->stream);
  cudaEventSynchronize(resampler->toc);
  if (aphids_ctx->iters % APHIDS_UPDATE_EVERY == 0) {
    cudaEventElapsedTime(&elapsedTime,resampler->tic,resampler->toc);
    aphids_log(aphids_ctx, APHIDS_LOG_INFO, "plan 2 cufft took %f ms [%f Gbps]",
      elapsedTime, resampler->batch[2]*resampler->fft_size[2] / (elapsedTime * 1e-3) /1e9 /2);
    fprintf(stdout,"GPU_DEBUG : plan 2 took %f ms [%f Gbps] \n", 
      elapsedTime, resampler->batch[2]*resampler->fft_size[2] / (elapsedTime * 1e-3) /1e9 /2);
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
  HANDLE_ERROR( cudaFree(resampler->beng_0) );
  HANDLE_ERROR( cudaFree(resampler->beng_1) );
  HANDLE_ERROR( cudaFree(resampler->beng_fluffed_0) );
  HANDLE_ERROR( cudaFree(resampler->beng_fluffed_1) );
  HANDLE_ERROR( cudaFree(resampler->swarm_c2r_ifft_out_0) );
  HANDLE_ERROR( cudaFree(resampler->swarm_c2r_ifft_out_1) );
  HANDLE_ERROR( cudaFree(resampler->swarm_r2c_fft_out_0) );
  HANDLE_ERROR( cudaFree(resampler->swarm_r2c_fft_out_1) );
  HANDLE_ERROR( cudaFree(resampler->r2dbe_c2r_ifft_out_0) );
  HANDLE_ERROR( cudaFree(resampler->r2dbe_c2r_ifft_out_1) );
  HANDLE_ERROR( cudaFree(resampler->gpu_out_buf) );
  
  for (i=0; i < 3; ++i){
    cufft_status = cufftDestroy(resampler->cufft_plan[i]);
    if (cufft_status != CUFFT_SUCCESS) {
//~ #ifdef GPU_DEBUG
          hashpipe_error(__FILE__, "CUFFT error: problem destroying plan %d\n", i);
//~ #endif
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
  int iter;
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

  // get the MHz trim parameter
  char *meta_str = (char *)malloc(sizeof(char)*16);  
  if (aphids_get(&aphids_ctx, "trim_from_dc", meta_str) != APHIDS_OK) {
    fprintf(stdout,"%s:%s(%d): could not read MHz trim width, using default\n",__FILE__,__FUNCTION__,__LINE__);
  }
  int mhz_trim = atoi(meta_str);
  fprintf(stdout,"%s:%s(%d): trimming %d MHz from DC\n",__FILE__,__FUNCTION__,__LINE__,mhz_trim);
  free(meta_str);

  /* Initialize GPU  */
  //~ fprintf(stdout, "sizeof(vdif_in_packet_block_t): %d\n", sizeof(vdif_in_packet_block_t));
  //~ fprintf(stdout, "sizeof(vdif_out_packet_block_t): %d\n", sizeof(vdif_out_packet_block_t));

  // initalize resampler
  resampler = (aphids_resampler_t *) malloc(NUM_GPU*sizeof(aphids_resampler_t));
  for (i=0; i< NUM_GPU; i++){
    aphids_resampler_init(&(resampler[i]), mhz_trim, i);
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
	// zero out the buffer first
	cudaMemset(resampler[i].beng_0, 0, BENG_FRAMES_PER_GROUP*BENG_SNAPSHOTS*BENG_CHANNELS_*2*sizeof(int8_t));
	cudaMemset(resampler[i].beng_1, 0, BENG_FRAMES_PER_GROUP*BENG_SNAPSHOTS*BENG_CHANNELS_*2*sizeof(int8_t));
	
	// call to vdif_to_beng?
	threads.x = 32; threads.y = 32; threads.z = 1;
	blocks.x = 128, blocks.y = 1; blocks.z = 1; 
	vdif_to_beng<<<blocks,threads>>>((int32_t*) this_bgc.bgv_buf_gpu, 
		resampler[i].beng_0, resampler[i].beng_1,
		this_bgc.beng_group_vdif_packet_count,b_zero);
	cudaDeviceSynchronize();
	
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
	
	for (iter=0; iter<RESAMPLE_BATCH_ITERATIONS; iter++) {
		// first zero output buffers
		cudaMemset((void *)resampler[i].beng_fluffed_0, 0, RESAMPLE_BATCH_SNAPSHOTS*BENG_CHANNELS*sizeof(cufftComplex));
		cudaMemset((void *)resampler[i].beng_fluffed_1, 0, RESAMPLE_BATCH_SNAPSHOTS*BENG_CHANNELS*sizeof(cufftComplex));
		
		// then inflate batch to cufftComplex
		threads.x = 32; threads.y = 32; threads.z = 1;
		blocks.x = 16, blocks.y = 1; blocks.z = 1;
		beng_int8_to_cufftComplex<<<blocks,threads>>>(
			resampler[i].beng_0 + iter*2*RESAMPLE_BATCH_SNAPSHOTS*BENG_CHANNELS_,
			resampler[i].beng_fluffed_0, RESAMPLE_BATCH_SNAPSHOTS);
		cudaDeviceSynchronize();
		beng_int8_to_cufftComplex<<<blocks,threads>>>(
			resampler[i].beng_1 + iter*2*RESAMPLE_BATCH_SNAPSHOTS*BENG_CHANNELS_,
			resampler[i].beng_fluffed_1, RESAMPLE_BATCH_SNAPSHOTS);
		cudaDeviceSynchronize();
		
		// transform SWARM spectra to time series
		state = SwarmC2R(&(resampler[i]), &aphids_ctx);
		cudaDeviceSynchronize();
		
		// transform SWARM time series to R2DBE compatible spectra
		state = SwarmR2C(&(resampler[i]), &aphids_ctx);
		cudaDeviceSynchronize();
		
		// transform R2DBE spectra to trimmed and resampled time series
		state = R2dbeC2R(&(resampler[i]), &aphids_ctx);
		cudaDeviceSynchronize();
		
		// calculate threshold for quantization
#ifdef QUANTIZE_THRESHOLD_COMPUTE
		if (resampler[i].quantizeThreshold_0 == 0) {
			// wrap as device pointer
			thrust::device_ptr<float> dev_ptr;
			// setup arguments
			summary_stats_unary_op<float>  unary_op;
			summary_stats_binary_op<float> binary_op;
			summary_stats_data<float>      init,result;
			// initialize
			init.initialize();
			// calculate quantization parameters for pol 0
			dev_ptr = thrust::device_pointer_cast((float *) resampler[i].r2dbe_c2r_ifft_out_0);
			result = thrust::transform_reduce(dev_ptr, 
				dev_ptr + FFT_BATCHES_R2DBE_C2R*FFT_SIZE_R2DBE_C2R,
				unary_op, init, binary_op);
			resampler[i].quantizeThreshold_0 = sqrt(result.variance());
			resampler[i].quantizeOffset_0 = result.average();
			// calculate quantization parameters for pol 1
			dev_ptr = thrust::device_pointer_cast((float *) resampler[i].r2dbe_c2r_ifft_out_1);
			init.initialize();
			result = thrust::transform_reduce(dev_ptr, 
				dev_ptr + FFT_BATCHES_R2DBE_C2R*FFT_SIZE_R2DBE_C2R,
				unary_op, init, binary_op);
			resampler[i].quantizeThreshold_1 = sqrt(result.variance());
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
		cudaDeviceSynchronize();
#endif
		// quantize to 2-bits
		threads.x = 16; threads.y = 32; threads.z = 1;
		blocks.x = 512; blocks.y = 1; blocks.z = 1;
		quantize2bit<<<blocks,threads>>>(
			(float *) resampler[i].r2dbe_c2r_ifft_out_0, // input
			(unsigned int*) &(resampler[i].gpu_out_buf->chan[0].datas[iter*VDIF_OUT_PKTS_PER_BLOCK/RESAMPLE_BATCH_ITERATIONS]),    // output
			FFT_BATCHES_R2DBE_C2R*FFT_SIZE_R2DBE_C2R,    // number of samples
			resampler[i].quantizeThreshold_0,            // threshold
			resampler[i].quantizeOffset_0                // offset
		);
		cudaDeviceSynchronize();
		quantize2bit<<<blocks,threads>>>(
			(float *) resampler[i].r2dbe_c2r_ifft_out_1, // input
			(unsigned int*) &(resampler[i].gpu_out_buf->chan[1].datas[iter*VDIF_OUT_PKTS_PER_BLOCK/RESAMPLE_BATCH_ITERATIONS]),    // output
			FFT_BATCHES_R2DBE_C2R*FFT_SIZE_R2DBE_C2R,    // number of samples
			resampler[i].quantizeThreshold_0,            // threshold
			resampler[i].quantizeOffset_0                // offset
		);
		cudaDeviceSynchronize();
	}

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
