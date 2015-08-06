#ifndef VDIF_IN_DATABUF_H
#define VDIF_IN_DATABUF_H

#ifndef STANDALONE_TEST
#include "hashpipe.h"
#include "hashpipe_databuf.h"
#endif // STANDALONE_TEST

////////////////////////////////////////////////////////////////////////
// This set of structures are used for memory storage of B-engine-over- 
// VDIF. The input thread will receive VDIF packets, pack them into a   
// buffer in memory according to the B-engine frames they belong to,    
// and eventually copy the buffer to GPU when necessary.                
////////////////////////////////////////////////////////////////////////
// Definition of VDIF header from m6support-0.14i (Mark6 software)
#define VDIF_IN_PKT_HEADER_SIZE 32 // size in bytes
typedef struct vdif_in_header {
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
	struct beng_hdr {
		uint32_t b_upper;
		uint8_t  c;
		uint8_t  z;
		uint8_t  f;
		uint8_t  b_lower;
	} beng;
	uint64_t edh_psn;
} vdif_in_header_t;

// Definition of VDIF packets, in this case VDIF-encapsulated B-engine
// packets as output by SDBE.
#define VDIF_IN_PKT_DATA_SIZE 1024 // size in bytes
typedef struct vdif_in_packet {
  vdif_in_header_t header;
  char data[VDIF_IN_PKT_DATA_SIZE];
} vdif_in_packet_t;

// This buffer holds enough VDIF packets to reconstruct an entire 
// B-engine frame, although not all packets may belong to the same
// B-engine counter or even have unique (f-id,ch-id) combinations. This
// structure is simply used for convenient memory size handling.
#define VDIF_PER_BENG_FRAME 2048
typedef struct beng_frame_vdif_buffer {
	vdif_in_packet_t vdif_packets[VDIF_PER_BENG_FRAME];
} beng_frame_vdif_buffer_t;

// This buffer holds enough VDIF packets to reconstruct a group of
// B-engine frames. The size of the group is determined by the specific
// data rate and format. 
// For different rates:
//   6/11 rate -- Preprocessing takes 39-sized units as input and 
//     delivers 32-sized units as output.
//   8/11 rate -- Preprocessing takes 13-sized units as input and
//     delivers 16-sized units as output.
// For different formats:
//   03/15 reordeing -- The March/July 2015 SDBE requires data reordering 
//     across adjacent B-engine frames, so that the size of input units 
//     should be larger by 1.
// Examples:
//   6/11 rate + 03/15 reordering = 39 + 1 = 40-sized units at input
//   8/11 rate + 03/15 reordering = 13 + 1 = 14-sized units at input
#define BENG_FRAMES_PER_GROUP 40 // 6/11 rate + 03/15 reordering
//~ #define BENG_FRAMES_PER_GROUP 14 // 8/11 rate + 03/15 reordering
typedef struct beng_group_vdif_buffer {
	beng_frame_vdif_buffer_t frames[BENG_FRAMES_PER_GROUP];
} beng_group_vdif_buffer_t;




////////////////////////////////////////////////////////////////////////
// This set of structures are used for communicating data between two
// hashpipe threads. Since the input thread collects VDIF frames that 
// belong to a given group of B-engine frames and then copies the 
// collection to GPU, it passes to the next thread a structure that:
//   * points to the GPU memory where the collection is stored
//   * indicates the completion of each B-engine frame in the group
////////////////////////////////////////////////////////////////////////
// Flag byte that indicates the completion of a single channel
#define FID_PER_CHAN 8 // number of f-ids for full channel
typedef struct channel_completion {
	uint8_t f0: 1;
	uint8_t f1: 1;
	uint8_t f2: 1;
	uint8_t f3: 1;
	uint8_t f4: 1;
	uint8_t f5: 1;
	uint8_t f6: 1;
	uint8_t f7: 1;
} channel_completion_t;

// Structure that indicates the completion of a single B-engine frame
#define CHAN_PER_BENG 256 // number of c-ids for full spectrum
typedef struct beng_frame_completion {
	uint64_t b; // B-engine counter value
	uint16_t beng_frame_vdif_packet_count; // Number of VDIF packets found for this B-engine counter
	channel_completion_t cc[CHAN_PER_BENG]; // Bit-array that flags found frames
} beng_frame_completion_t;

// Structure that encapsulates a single data unit passed between 
// hashpipe threads.
typedef struct beng_group_completion {
	int32_t beng_group_vdif_packet_count; // Number of VDIF packets found for this group of B-engine counters
	beng_group_vdif_buffer_t *bgv_buf_cpu; // Pointer to CPU buffer filled with VDIF packets
	beng_group_vdif_buffer_t *bgv_buf_gpu; // Pointer to GPU buffer filled with VDIF packets
	beng_frame_completion_t bfc[BENG_FRAMES_PER_GROUP];
} beng_group_completion_t;

// The buffer holding a given number of data units passed between 
// hashpipe threads.
#define BENG_GROUPS_IN_BUFFER 4
typedef struct vdif_in_databuf {
#ifndef STANDALONE_TEST
  hashpipe_databuf_t header;
#endif // STANDALONE_TEST
  beng_group_completion_t bgc[BENG_GROUPS_IN_BUFFER];
} vdif_in_databuf_t;

#ifndef STANDALONE_TEST
// Creating hashpipe data buffer
hashpipe_databuf_t *vdif_in_databuf_create(int instance_id, int databuf_id);
#endif // STANDALONE_TEST

// Read B-engine counter value from the given VDIF header
// Arguments:
// ----------
//   vdif_pkt_hdr -- Pointer to memory where VDIF header is located
// Return:
// -------
//   The B-engine counter value embedded in the VDIF header
// Notes:
// ------
//   The returned value is int64_t, which is large enough to store
//   40-bit wide field in the VDIF header. A signed type is useful so
//   that negative values can be used as error or other indicators, 
//   while the true counter value will always be semi-positive.
int64_t get_packet_b_count(vdif_in_header_t *vdif_pkt_hdr);

// Initialize a single data unit, beng_group_completion_t
// Arguments:
// ----------
//   bgc         -- Pointer to the data unit to be initialized
//   bgv_buf_cpu -- Pointer to memory where VDIF data for this group are 
//     stored in CPU accessible memory.
//   bgv_buf_gpu -- Pointer to GPU memory where VDIF data will be 
//     transferred to eventually.
//   b_start     -- Lowest B-engine counter associated with this group
// Return:
// -------
//   void
// Notes:
// ------
//   The CPU buffer is used to collect VDIF data belonging to a until 
//   particular group of B-engine frames. Once the group is ready to be
//   processed, the data in the CPU buffer is copied to the GPU buffer
//   and the assoicated hashpipe buffer index marked filled so that the
//   next hashpipe thread can start processing on it.
void init_beng_group(beng_group_completion_t *bgc, beng_group_vdif_buffer_t *bgv_buf_cpu, beng_group_vdif_buffer_t *bgv_buf_gpu, int64_t b_start);

// Get B-engine frame group index relative to current first for given 
// VDIF packet
// Arguments:
// ----------
//   bgc_buf   -- Pointer to (hashpipe) buffer containing all data units
//   index_ref -- Index of current first data unit (one who's B-engine 
//     counter range has the lowest start value)
//   vdif_pkt  -- The VDIF packet to be assigned to a group
// Return:
// -------
//   The offset relative to the reference index where the given VDIF 
//   packet will be stored
// Notes:
// ------
//   Reference index will point to the currently-being-filled hashpipe
//   buffer index, and the VDIF insertion index can be calcualted as:
//     index_ins = (index_ref + index_off) % N_index
//   where index_off is the value returned by this function and N_index 
//   is the size of the hashpipe buffer.
int get_beng_group_index_offset(vdif_in_databuf_t *bgc_buf, int index_ref, vdif_in_packet_t *vdif_pkt);

// Insert VDIF packet into group buffer
// Arguments:
// ----------
//   bgc_buf   -- Pointer to (hashpipe) buffer containing all data units
//   index_ref -- Index of current first data unit (one who's B-engine 
//     counter range has the lowest start value)
//   offset    -- Offset returned by get_beng_group_index_offset
//   vdif_pkt  -- The VDIF packet to be assigned to a group
// Return:
// -------
//   
int insert_vdif_in_beng_group_buffer(vdif_in_databuf_t *bgc_buf, int index_ref, int offset, vdif_in_packet_t *vdif_pkt);
int check_beng_group_complete(vdif_in_databuf_t *bgc_buf, int index);
int transfer_beng_group_to_gpu(vdif_in_databuf_t *bgc_buf, int index);
int check_transfer_complete(int index);

#endif // VDIF_IN_DATABUF_H
