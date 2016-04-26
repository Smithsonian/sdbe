#ifndef VDIF_2PAC_H
#define VDIF_2PAC_H

#include "vdif_in_databuf.h"

#define VTP_BYTE_SIZE 8
typedef struct vdif_2pac_packet {
  vdif_in_packet_t pkt1;
  char vtp[VTP_BYTE_SIZE];
  vdif_in_packet_t pkt2;
} vdif_2pac_packet_t;

void unpack_2pac(vdif_in_packet_t *dest, vdif_2pac_packet_t *src, int n_src);

#endif // VDIF_2PAC_H
