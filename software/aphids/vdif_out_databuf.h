#ifndef VDIF_OUT_DATABUF_H
#define VDIF_OUT_DATABUF_H

#include <cuda_runtime.h>

#include "hashpipe.h"
#include "hashpipe_databuf.h"

////////////////////////////////////////////////////////////////////////
// This set of structures are used for memory storage of normal VDIF on
// the output side before transmitting.
////////////////////////////////////////////////////////////////////////
// Definition of VDIF header from m6support-0.14i (Mark6 software)
#define VDIF_OUT_PKT_HEADER_SIZE 32 // size in bytes
typedef struct vdif_out_header {
	struct word0 {
		uint32_t secs_inre:30;
		uint32_t legacy:1;
		uint32_t invalid:1;
	} w0;
	struct word1 {
		uint32_t df_num_insec:24;
		uint32_t ref_epoch:6;
		uint32_t UA:2;
	} w1;
	struct word2 {
		uint32_t df_len:24;
		uint32_t num_channels:5;
		uint32_t ver:3;
	} w2;
	struct word3 {
		uint32_t stationID:16;
		uint32_t threadID:10;
		uint32_t bps:5;
		uint32_t dt:1;
	} w3;
	struct word4 {
		uint32_t eud5:24;
		uint32_t edv5:8;
	} w4;
	struct word5 {
		uint32_t PICstatus;
	} w5;
	uint64_t edh_psn;
} vdif_out_header_t;

// Definition of VDIF packets, in this case the usual VDIF to be sent
// over high-speed network. The recommended data size is that which 
// makes the timespan a single packet equivalent to that for the R2DBE 
// data stream, i.e. 8us per packet. For 2bit samples over half the 
// bandwidth (2 SWARM channels), the packet size is 4096 bytes.
#define VDIF_OUT_PKT_DATA_SIZE 4096 // size in bytes
typedef struct vdif_out_packet {
  vdif_out_header_t header;
  char data[VDIF_OUT_PKT_DATA_SIZE];
} vdif_out_packet_t;

// This buffer should be large enough to hold the maximum expected 
// number of VDIF packets per block of input data, which is dependent
// on:
//   * The number of B-engine frames processed per group at the input
//   * The bit depth of the output samples
//   * The payload size at the output
// For:
//   *  39 B-engine frames (6/11 rate), 4096Byte output packet size, 
//      2bit output samples: 16384 VDIF packets per B-engine group.
//      39 x 128 x 32768 = 163577856 samples at 6/11 SWARM rate
//      163577856 * 4096 / 2496 = 268435456 samples at R2DBE rate
//      268435456 / (4096Bytes / 0.25samples-per-Byte) = 16384 packets
// This holds VDIF for single SWARM channel, so typicall need two of 
// these (in parallel) for all the data
#define VDIF_OUT_PKTS_PER_BLOCK 16384
typedef struct vdif_out_packet_block {
  vdif_out_packet_t packets[VDIF_OUT_PKTS_PER_BLOCK];
} vdif_out_packet_block_t;




////////////////////////////////////////////////////////////////////////
// This set of structures are used for communicating data between two
// hashpipe threads. The actual data received will just be a very large
// array of X-bit samples packet into 32bit wide words, and the data
// passed through the hashpipe buffer should contain
//   * handle to shared GPU memory where data is stored
//   * metadata that describes the format and size of stored data
////////////////////////////////////////////////////////////////////////
// Structure that encapsulates a single data unit passed between 
// hashpipe threads.
typedef struct quantized_storage {
	int bit_depth; // bit depth, not always 2bit?
	int N_32bit_words_chan0; // size of quantized storage measured in 32bit wide units, channel 0 data
	int N_32bit_words_chan1; // size of quantized storage measured in 32bit wide units, channel 1 data
	cudaIpcMemHandle_t ipc_mem_handle_chan0; // used for address translation in separate process, channel 0 data
	cudaIpcMemHandle_t ipc_mem_handle_chan1; // used for address translation in separate process, channel 1 data
} quantized_storage_t;

// The buffer holding a given number of data units passed between 
// hashpipe threads.
#define VDIF_OUT_BUFFER_SIZE 4
typedef struct vdif_out_databuf {
  hashpipe_databuf_t header;
  quantized_storage_t blocks[VDIF_OUT_BUFFER_SIZE];
} vdif_out_databuf_t;

hashpipe_databuf_t *vdif_out_databuf_create(int instance_id, int databuf_id);

#endif
