#include <string.h>

#include "vdif_8pac.h"

void reset_8pac_size(vdif_8pac_packet_t *pkt) {
	int ii;
	uint32_t newsize;
	newsize = ((pkt->pkt1.header.w2.df_len-7)/8) & 0x00FFFFFF;
	pkt->pkt1.header.w2.df_len = newsize;
	for (ii=0; ii<7; ii++) {
		pkt->pac7[ii].pac.header.w2.df_len = newsize;
	}
}

void fix_8pac_data_frame(vdif_8pac_packet_t *pkt) {
	uint32_t df8;
	int ii;
	df8 = 8*pkt->pkt1.header.w1.df_num_insec;
	pkt->pkt1.header.w1.df_num_insec = df8;
	for (ii=0; ii<7; ii++) {
		pkt->pac7[ii].pac.header.w1.df_num_insec = df8 + ii+1;
	}
}

void fix_8pac_psn(vdif_8pac_packet_t *pkt) {
	uint64_t psn8;
	int ii;
	int idx_inc = -1;
	psn8 = 8*pkt->pkt1.header.edh_psn;
	pkt->pkt1.header.edh_psn = psn8;
	for (ii=0; ii<7; ii++) {
		pkt->pac7[ii].pac.header.edh_psn = psn8 + ii+1;
	}
}

void fix_8pac_flags(vdif_8pac_packet_t *pkt) {
	int ii;
	for (ii=0; ii<7; ii++) {
		pkt->pac7[ii].pac.header.w0.invalid = pkt->pkt1.header.w0.invalid;
	}
}

void unpack_8pac(vdif_in_packet_t *dest, vdif_8pac_packet_t *src, int n_src) {
	int aightpac_count = 0, vdif_count = 0;
	int ii;
	for (aightpac_count=0; aightpac_count<n_src; aightpac_count++) {
		reset_8pac_size(src+aightpac_count);
		fix_8pac_data_frame(src+aightpac_count);
		fix_8pac_psn(src+aightpac_count);
		fix_8pac_flags(src+aightpac_count);
		memcpy(dest+vdif_count++,&((src+aightpac_count)->pkt1),sizeof(vdif_in_packet_t));
		for (ii=0; ii<7; ii++) { 
			memcpy(dest+vdif_count++,&((src+aightpac_count)->pac7[ii].pac),sizeof(vdif_in_packet_t));
		}
	}
}
