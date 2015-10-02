#ifndef SGCOMM_BENG_OVER_VDIF_H
#define SGCOMM_BENG_OVER_VDIF_H

// *********************************************************************
// These definitions are used to do time realignment on day 85, 2015 data
// bitfield that defines valid FIDs usable for b_first inference
#define DAY085_FID_BFIRST (0xFC) // all but 0,1
// bitfield that defines valid FIDs usable for VDIF template inference
#define DAY085_FID_VDIF_TEMPLATE (0x03) // only 0,1
// *********************************************************************

#define VDIF_IN_PKT_HEADER_SIZE 32 // size in bytes
typedef struct vdif_in_header {
	struct word0_in {
		uint32_t secs_inre:30;
		uint32_t legacy:1;
		uint32_t invalid:1;
	} w0;
	struct word1_in {
		uint32_t df_num_insec:24;
		uint32_t ref_epoch:6;
		uint32_t UA:2;
	} w1;
	struct word2_in {
		uint32_t df_len:24;
		uint32_t num_channels:5;
		uint32_t ver:3;
	} w2;
	struct word3_in {
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

int64_t get_packet_b_count(vdif_in_header_t *vdif_pkt_hdr);

#endif // SGCOMM_BENG_OVER_VDIF_H
