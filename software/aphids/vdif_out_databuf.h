#ifndef VDIF_OUT_DATABUF_H
#define VDIF_OUT_DATABUF_H

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#define VDIF_OUT_PKT_HEADER_SIZE 32
#define VDIF_OUT_PKT_DATA_SIZE 4096
#define VDIF_OUT_PKTS_PER_BLOCK 16
#define VDIF_OUT_BUFFER_SIZE 8

typedef struct vdif_out_packet {
  char header[VDIF_OUT_PKT_HEADER_SIZE];
  char data[VDIF_OUT_PKT_DATA_SIZE];
} vdif_out_packet_t;

typedef struct vdif_out_packet_block {
  vdif_out_packet_t packets[VDIF_OUT_PKTS_PER_BLOCK];
} vdif_out_packet_block_t;

typedef struct vdif_out_databuf {
  hashpipe_databuf_t header;
  vdif_out_packet_block_t blocks[VDIF_OUT_BUFFER_SIZE];
} vdif_out_databuf_t;

hashpipe_databuf_t *vdif_out_databuf_create(int instance_id, int databuf_id);

#endif
