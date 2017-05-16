#ifndef VDIF_INOUT_GPU_THREAD_H
#define VDIF_INOUT_GPU_THREAD_H

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "sample_rates.h"

// VDIF constants
#define VDIF_BYTE_SIZE 1056 // VDIF frame size in bytes
#define VDIF_BYTE_SIZE_HEADER 32 // VDIF header size in bytes
#define VDIF_BYTE_SIZE_DATA 1024 // VDIF data size in bytes
#define VDIF_INT_SIZE (1056/4) // VDIF frame size in int
#define VDIF_INT_SIZE_HEADER (32/4) // VDIF header size in int
#define VDIF_INT_SIZE_DATA (1024/4) // VDIF data size in int
#define VDIF_BIT_DEPTH 2 // bits-per-sample
#define VDIF_PER_BENG 2048

// Data structure
#define BENG_CHANNELS_ 16384
#define BENG_CHANNELS (BENG_CHANNELS_+1) // number of channels PLUS added sample-rate/2 component for the complex-to-real inverse transform
#define BENG_SNAPSHOTS 128
#define SWARM_N_FIDS 8
#define SWARM_XENG_PARALLEL_CHAN 8

//////////////////////// Resampling ////////////////////////////////////
/* Number of snapshots to batch process at a time, should be an integer
 * multiple of DECIMATION_FACTOR and smaller than
 * BENG_SNAPSHOTS*BENG_FRAMES_PER_GROUP.
 */
#define RESAMPLE_BATCH_SNAPSHOTS DECIMATION_FACTOR
// number of iterations needed to process entire B
#define RESAMPLE_BATCH_ITERATIONS (BENG_FRAMES_PER_GROUP * BENG_SNAPSHOTS / RESAMPLE_BATCH_SNAPSHOTS)
// SWARM_C2R FFT size
#define FFT_SIZE_SWARM_C2R 2*BENG_CHANNELS_
// SWARM_C2R FFT batches
#define FFT_BATCHES_SWARM_C2R RESAMPLE_BATCH_SNAPSHOTS
/* The resampling FFT+IFFT pair should do at least a DECIMATION_FACTOR-
 * sized FFT and EXPANSION_FACTOR-sized IFFT. Increase the FFT sizes by
 * the factor FFT_SIZE_SCALE. This value has to be a multiple of 32,
 * since that will ensure that an integer number of MHz channels can be
 * peeled off on either sides to reduce the SWARM full rate to the R2DBE
 * full rate.
 */
#define FFT_SIZE_SWARM_R2C_R2DBE_C2R_SCALE 256
// SWARM_R2C FFT size
#define FFT_SIZE_SWARM_R2C (DECIMATION_FACTOR*FFT_SIZE_SWARM_R2C_R2DBE_C2R_SCALE)
// SWARM_R2C FFT batches
#define FFT_BATCHES_SWARM_R2C (RESAMPLE_BATCH_SNAPSHOTS * (2*BENG_CHANNELS_) / FFT_SIZE_SWARM_R2C)
// R2DBE_C2R IFFT size
#define FFT_SIZE_R2DBE_C2R (EXPANSION_FACTOR*FFT_SIZE_SWARM_R2C_R2DBE_C2R_SCALE)
// R2DBE_C2R IFFT batches, should be equal to FFT_BATCHES_SWARM_R2C
#define FFT_BATCHES_R2DBE_C2R FFT_BATCHES_SWARM_R2C

//output
#define OUTPUT_MAX_VALUE_MASK_2BIT 3
#define OUTPUT_MAX_VALUE_MASK_4BIT 15

// VDIF packed B-engine packet
#define BENG_VDIF_HDR_0_OFFSET_INT 4 // b1 b2 b3 b4
#define BENG_VDIF_HDR_1_OFFSET_INT 5 //  c  0  f b0
#define BENG_VDIF_CHANNELS_PER_INT 4 // a-d in a single int32_t, and e-h in a single int32_t
#define BENG_VDIF_INT_PER_SNAPSHOT (SWARM_XENG_PARALLEL_CHAN/BENG_VDIF_CHANNELS_PER_INT)
#define BENG_PACKETS_PER_FRAME (BENG_CHANNELS_/SWARM_XENG_PARALLEL_CHAN)
#define BENG_FRAME_COMPLETION_COMPLETE_ON_GPU (BENG_PACKETS_PER_FRAME*blockDim.x) // value of completion counter when B-engine frame complete, multiplication by THREADS_PER_BLOCK_x required since all x-threads increment counter
#define BENG_FRAME_COMPLETION_COMPLETE_ON_CPU (BENG_PACKETS_PER_FRAME*num_x_threads) // value of completion counter when B-engine frame complete, multiplication by THREADS_PER_BLOCK_x required since all x-threads increment counter
#define BENG_VDIF_SAMPLE_VALUE_OFFSET 2.0f

#endif // VDIF_INOUT_GPU_THREAD


