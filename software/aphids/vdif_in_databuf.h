#ifndef VDIF_IN_DATABUF_H
#define VDIF_IN_DATABUF_H

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#define VDIF_IN_PKT_HEADER_SIZE 32
#define VDIF_IN_PKT_DATA_SIZE 1024
#define VDIF_IN_PKTS_PER_BLOCK 16
#define VDIF_IN_BUFFER_SIZE 8

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
        uint64_t b:40;
        uint64_t f:8;
        uint64_t z:8;
        uint64_t c:8;
    } beng;
    uint64_t edh_psn;
} vdif_in_header_t;

typedef struct vdif_in_packet {
  vdif_in_header_t header;
  char data[VDIF_IN_PKT_DATA_SIZE];
} vdif_in_packet_t;

typedef struct vdif_in_packet_block {
  vdif_in_packet_t packets[VDIF_IN_PKTS_PER_BLOCK];
} vdif_in_packet_block_t;

typedef struct vdif_in_databuf {
  hashpipe_databuf_t header;
  vdif_in_packet_block_t blocks[VDIF_IN_BUFFER_SIZE];
} vdif_in_databuf_t;

hashpipe_databuf_t *vdif_in_databuf_create(int instance_id, int databuf_id);

#endif
