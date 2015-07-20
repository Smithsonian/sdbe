#ifndef BENG_VDIF_BUFFER_H
#define BENG_VDIF_BUFFER_H

#define FID_PER_CHAN 8
#define CHAN_PER_BENG 256
#define VDIF_PER_BENG_FRAME 2048
#define BENG_FRAMES_PER_GROUP 40
#define BENG_GROUPS_IN_BUFFER 4

typedef struct beng_frame_vdif_buffer {
	vdif_in_packet_t vdif_packets[VDIF_PER_BENG_FRAME];
} beng_frame_vdif_buffer_t;

typedef struct beng_group_vdif_buffer {
	beng_frame_vdif_buffer_t frames[BENG_FRAMES_PER_GROUP];
} beng_group_vdif_buffer_t;

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

typedef struct beng_frame_completion {
	uint64_t b; // B-engine counter value
	uint16_t beng_frame_vdif_packet_count; // Number of VDIF packets found for this B-engine counter
	channel_completion_t cc[CHAN_PER_BENG]; // Bit-array that flags found frames
} beng_frame_completion_t;

typedef struct beng_group_completion {
	int32_t beng_group_vdif_packet_count; // Number of VDIF packets found for this group of B-engine counters
	beng_group_vdif_buffer_t *bgv_buf; // Pointer to buffer filled with VDIF packets
	beng_frame_completion_t bfc[BENG_FRAMES_PER_GROUP];
} beng_group_completion_t;

typedef struct beng_group_completion_buffer {
	beng_group_completion_t bgc[BENG_GROUPS_IN_BUFFER];
} beng_group_completion_buffer_t;

void init_beng_group(beng_group_completion_t *bgc, beng_group_vdif_buffer_t *bgv_buf, int64_t b_start);
int get_beng_group_index_offset(beng_group_completion_buffer_t *bgc_buf, int index_ref, vdif_in_packet_t *vdif_pkt);
int insert_vdif_in_beng_group_buffer(beng_group_completion_buffer_t *bgc_buf, int index_ref, int offset, vdif_in_packet_t *vdif_pkt);
int check_beng_group_complete(beng_group_completion_buffer_t *bgc_buf,int index);

#endif // BENG_VDIF_BUFFER_H
