#ifndef VDIF_UPLOAD_TO_GPU_DATABUF_H
#define VDIF_UPLOAD_TO_GPU_DATABUF_H

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#define FID_PER_CHAN 8
#define CHAN_PER_BENG 256
#define VDIF_PER_BENG_FRAME 2048
#define BENG_FRAMES_PER_BLOCK 40
#define VDIF_UPLOAD_TO_GPU_BUFFER_SIZE 4

typedef struct beng_frame_vdif {
	vdif_in_packet_t vdif_packets[VDIF_PER_BENG_FRAME];
} beng_frame_vdif_t;

typedef struct beng_frame_vdif_buffer {
	beng_frame_vdif_t frames[BENG_FRAMES_PER_BLOCK];
} beng_frame_vdif_buffer_t;

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

typedef struct beng_completion {
	uint64_t b; // B-engine counter value
	uint16_t vdif_packet_count; // Number of VDIF packets found for this B-engine counter
	channel_completion_t ch_comp[CHAN_PER_BENG]; // Bit-array that flags found frames
} beng_completion_t;

typedef struct vdif_upload_to_gpu_block {
	beng_frame_vdif_buffer_t *gpu_beng_vdif_buf;
	beng_completion_t b_comp[BENG_FRAMES_PER_BLOCK];
} vdif_upload_to_gpu_block_t;

typedef struct vdif_upload_to_gpu_databuf {
	hashpipe_databuf_t header;
	vdif_upload_to_gpu_block_t blocks[VDIF_UPLOAD_TO_GPU_BUFFER_SIZE];
} vdif_upload_to_gpu_databuf_t;

hashpipe_databuf_t *vdif_upload_to_gpu_databuf_create(int instance_id, int databuf_id);

#endif // VDIF_UPLOAD_TO_GPU_DATABUF_H
