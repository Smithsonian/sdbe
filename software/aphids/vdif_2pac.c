#include <string.h>

#include "vdif_2pac.h"

void reset_2pac_size(vdif_2pac_packet_t *pkt) {
	pkt->pkt1.header.w2.df_len = (pkt->pkt1.header.w2.df_len-1)/2;
	pkt->pkt2.header.w2.df_len = (pkt->pkt2.header.w2.df_len-1)/2;
}

void fix_2pac_data_frame(vdif_2pac_packet_t *pkt) {
	uint32_t df1,df2;
	df1 = pkt->pkt1.header.w1.df_num_insec;
	df2 = pkt->pkt2.header.w1.df_num_insec;
	pkt->pkt1.header.w1.df_num_insec = 2*df1;
	if (df2 > df1) {
		pkt->pkt2.header.w1.df_num_insec = 2*df2-1;
	} else {
		pkt->pkt2.header.w1.df_num_insec = 2*df2+1;
	}
}

void fix_2pac_psn(vdif_2pac_packet_t *pkt) {
	uint64_t psn1,psn2;
	psn1 = pkt->pkt1.header.edh_psn;
	psn2 = pkt->pkt2.header.edh_psn;
	pkt->pkt1.header.edh_psn = 2*psn1;
	if (psn2 > psn1) {
		pkt->pkt2.header.edh_psn = 2*psn2-1;
	} else {
		pkt->pkt2.header.edh_psn = 2*psn2+1;
	}
}

void fix_2pac_flags(vdif_2pac_packet_t *pkt) {
	pkt->pkt2.header.w0.invalid = pkt->pkt1.header.w0.invalid;
}

void unpack_2pac(vdif_in_packet_t *dest, vdif_2pac_packet_t *src, int n_src) {
	int tupac_count = 0, vdif_count = 0;
	for (tupac_count=0; tupac_count<n_src; tupac_count++) {
		reset_2pac_size(src+tupac_count);
		fix_2pac_data_frame(src+tupac_count);
		fix_2pac_psn(src+tupac_count);
		fix_2pac_flags(src+tupac_count);
		memcpy(dest+vdif_count++,&((src+tupac_count)->pkt1),sizeof(vdif_in_packet_t));
		memcpy(dest+vdif_count++,&((src+tupac_count)->pkt2),sizeof(vdif_in_packet_t));
	}
}
