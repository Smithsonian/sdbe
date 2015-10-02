#include "sgcomm_beng_over_vdif.h"

int64_t get_packet_b_count(vdif_in_header_t *vdif_pkt_hdr) {
	int64_t b = 0;
	b |= ((int64_t)(vdif_pkt_hdr->beng.b_upper)&(int64_t)0x00000000FFFFFFFF) << 8;
	b |= (int64_t)(vdif_pkt_hdr->beng.b_lower)&(int64_t)0x00000000000000FF;
	return b;
}
