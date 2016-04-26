#include <string.h>

#include "vdif_2pac.h"

void reset_2pac_size(vdif_2pac_packet_t *pkt) {
	pkt->pkt1.header.w2.df_len = (pkt->pkt1.header.w2.df_len-1)/2;
	pkt->pkt2.header.w2.df_len = (pkt->pkt2.header.w2.df_len-1)/2;
}

void unpack_2pac(vdif_in_packet_t *dest, vdif_2pac_packet_t *src, int n_src) {
	int tupac_count = 0, vdif_count = 0;
	for (tupac_count=0; tupac_count<n_src; tupac_count++) {
		reset_2pac_size(src+tupac_count);
		memcpy(dest+vdif_count++,&((src+tupac_count)->pkt1),sizeof(vdif_in_packet_t));
		memcpy(dest+vdif_count++,&((src+tupac_count)->pkt2),sizeof(vdif_in_packet_t));
	}
}
