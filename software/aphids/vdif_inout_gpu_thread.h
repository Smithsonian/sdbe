#ifndef VDIF_INOUT_GPU_THREAD_H
#define VDIF_INOUT_GPU_THREAD_H

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "sample_rates.h"

// rates
//~ #define SWARM_RATE 2496e6 <<--- definition moved to sample_rates.h
#define R2DBE_RATE 4096e6

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
//~ #define BENG_BUFFER_IN_COUNTS 40 // will likey be replaced by BENG_FRAMES_PER_GROUP <<--- definition moved to sample_rates.h
#define SWARM_N_FIDS 8
#define SWARM_XENG_PARALLEL_CHAN 8

#define UNPACKED_BENG_CHANNELS 16400

//output
#define OUTPUT_MAX_VALUE_MASK 3 // ((int)pow((double) 2,OUTPUT_BITS_PER_SAMPLE) - 1)

// Resampling factors
// Note that you need to check this plan.
//~ #define DECIMATION_FACTOR 39  <<--- definition moved to sample_rates.h
//~ #define EXPANSION_FACTOR 32  <<--- definition moved to sample_rates.h
#define RESAMPLING_CHUNK_SIZE (DECIMATION_FACTOR * 512)
#define RESAMPLING_BATCH 512

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


