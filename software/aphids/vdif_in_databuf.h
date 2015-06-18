#ifndef VDIF_IN_DATABUF_H
#define VDIF_IN_DATABUF_H

#include <pthread.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#define VDIF_PKT_HEADER_SIZE 32
#define VDIF_PKT_DATA_SIZE 4096
#define VDIF_IN_BUFFER_SIZE 32

typedef struct vdif_packet {
  char header[VDIF_PKT_HEADER_SIZE];
  char data[VDIF_PKT_DATA_SIZE];
} vdif_packet_t;

typedef struct vdif_in_databuf {
  hashpipe_databuf_t header;
  vdif_packet_t packets[VDIF_IN_BUFFER_SIZE];
} vdif_in_databuf_t;

hashpipe_databuf_t *vdif_in_databuf_create(int instance_id, int databuf_id);

#endif
