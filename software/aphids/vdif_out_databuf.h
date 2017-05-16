#ifndef VDIF_OUT_DATABUF_H
#define VDIF_OUT_DATABUF_H

#include <cuda_runtime.h>

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "sample_rates.h"

#define SWARM_ZERO_DELAY_OFFSET_PICOSECONDS 3715791

////////////////////////////////////////////////////////////////////////
// This set of structures are used for memory storage of normal VDIF on
// the output side before transmitting.
////////////////////////////////////////////////////////////////////////
// Definition of VDIF header from m6support-0.14i (Mark6 software)
#define VDIF_OUT_PKT_HEADER_SIZE 32 // size in bytes
typedef struct vdif_out_header {
	struct word0_out {
		uint32_t secs_inre:30;
		uint32_t legacy:1;
		uint32_t invalid:1;
	} w0;
	struct word1_out {
		uint32_t df_num_insec:24;
		uint32_t ref_epoch:6;
		uint32_t UA:2;
	} w1;
	struct word2_out {
		uint32_t df_len:24;
		uint32_t num_channels:5;
		uint32_t ver:3;
	} w2;
	struct word3_out {
		uint32_t stationID:16;
		uint32_t threadID:10;
		uint32_t bps:5;
		uint32_t dt:1;
	} w3;
	struct word4_out {
		uint32_t pol:1; // 0 = L/X, 1 = R/Y
		uint32_t bdc_sideband:1; // 0 = LSB, 1 = USB
		uint32_t rx_sideband:1; // 0 = LSB, 1 = USB
		uint32_t eud5:21;
		uint32_t edv5:8;
	} w4;
	struct word5_out {
		uint32_t PICstatus;
	} w5;
	uint64_t edh_psn;
} vdif_out_header_t;

// Definition of VDIF packets, in this case the usual VDIF to be sent
// over high-speed network. The recommended data size is that which 
// makes the timespan a single packet equivalent to that for the R2DBE 
// data stream, i.e. 8us per packet. For 2bit samples over half the 
// bandwidth (2 SWARM channels), the packet size is 4096 bytes.
// For 2bit samples over full bandwidth, the packet size is 8192 bytes 
// (valid for SWARM rates of 10/11 and higher).
// For 4bit samples over full bandwidth the packet size is 16384 bytes.
#define VDIF_OUT_PKT_DATA_SIZE 16384 // size in bytes
typedef struct vdif_out_data {
	char data[VDIF_OUT_PKT_DATA_SIZE];
} vdif_out_data_t;

typedef struct vdif_out_packet {
	vdif_out_header_t header;
	vdif_out_data_t data;
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
//      163577856 * 2048 / 2496 = 134217728 samples at R2DBE rate
//      134217728 / (4096Bytes / 0.25samples-per-Byte) = 8192 packets
//   *  13 B-engine frames (8/11 rate), 4096Byte output packet size, 
//      2bit output samples: 4096 VDIF packets per B-engine group.
//      13 x 128 x 32768 = 54525952 samples at 8/11 SWARM rate
//      54525952 * 2048 / 3328 = 33554432 samples at R2DBE rate
//      33554432 / (4096Bytes / 0.25samples-per-Byte) = 2048 packets
//   *  65 B-engine frames (10/11 rate), 8192Byte output packet size,
//      2bit output samples: 16384 VDIF packets per B-engine group.
//      65 x 128 x 32768 = 272629760 samples at 10/11 SWARM rate
//      272629760 * 4096 / 4160 = 268435456 samples at R2DBE rate
//      268435456 / (8192Bytes / 0.25samples-per-Byte) = 8192 packets
// This holds VDIF for single SWARM channel, so typicall need two of 
// these (in parallel) for all the data
//~ #define VDIF_OUT_PKTS_PER_BLOCK 8192 <<--- moved to sample_rates.h
typedef struct vdif_out_packet_block {
	vdif_out_packet_t packets[VDIF_OUT_PKTS_PER_BLOCK];
} vdif_out_packet_block_t;

// 
#define VDIF_CHAN 2
typedef struct vdif_out_packet_group {
	vdif_out_packet_block_t chan[VDIF_CHAN];
} vdif_out_packet_group_t;

////////////////////////////////////////////////////////////////////////
// This set of structures define raw data blocks, like the above but
// without any header storage.
////////////////////////////////////////////////////////////////////////
typedef struct vdif_out_data_block {
	vdif_out_data_t datas[VDIF_OUT_PKTS_PER_BLOCK];
} vdif_out_data_block_t;

typedef struct vdif_out_data_group {
	vdif_out_data_block_t chan[VDIF_CHAN];
} vdif_out_data_group_t;

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
	vdif_out_data_group_t *vdg_buf_cpu;
	vdif_out_data_group_t *vdg_buf_gpu;
	int bit_depth; // bit depth, not always 2bit?
	int N_32bit_words_per_chan; // size of quantized storage measured in 32bit wide units
	int gpu_id;
	void *memcpy_stream;
	vdif_out_header_t vdif_header_template;
} quantized_storage_t;

// The buffer holding a given number of data units passed between 
// hashpipe threads.
#define VDIF_OUT_BUFFER_SIZE 4
typedef struct vdif_out_databuf {
  hashpipe_databuf_t header;
  quantized_storage_t blocks[VDIF_OUT_BUFFER_SIZE];
} vdif_out_databuf_t;

hashpipe_databuf_t *vdif_out_databuf_create(int instance_id, int databuf_id);

int get_vpg_cpu_memory(vdif_out_packet_group_t **vpg_buf_cpu, int index);

int get_vdg_cpu_memory(vdif_out_data_group_t **vdg_buf_cpu, int index);

int transfer_vdif_group_to_cpu(vdif_out_databuf_t *qs_buf, int index);

int check_transfer_vdif_group_to_cpu_complete(vdif_out_databuf_t *qs_buf, int index);

void init_vdif_out(vdif_out_packet_group_t **vpg_buf_cpu, int index);

void print_quantized_storage(quantized_storage_t *qs, const char *tag);

#endif // VDIF_OUT_DATABUF_H
