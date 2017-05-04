#ifndef VDIF_8PAC_H
#define VDIF_8PAC_H

#include "vdif_in_databuf.h"

#define VTP_BYTE_SIZE 8

typedef struct vtp_plus_pac {
  char vtp[VTP_BYTE_SIZE];
  vdif_in_packet_t pac;
} vtp_plus_pac_t;

typedef struct vdif_8pac_packet {
  vdif_in_packet_t pkt1;
  vtp_plus_pac_t pac7[7];
} vdif_8pac_packet_t;

void unpack_8pac(vdif_in_packet_t *dest, vdif_8pac_packet_t *src, int n_src);

#endif // VDIF_8PAC_H
