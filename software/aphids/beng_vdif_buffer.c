#include <string.h>

// compilation options, for testing and debugging only
#define PLATFORM_CPU 0
#define PLATFORM_GPU 1
#define PLATFORM PLATFORM_CPU

#if PLATFORM == PLATFORM_GPU
	#include <cuda_runtime.h>
#endif

#include "vdif_in_databuf.h"
#include "beng_vdif_buffer.h"

void init_beng_group(beng_group_completion_t *bgc, beng_group_vdif_buffer_t *bgv_buf, int64_t b_start) {
	int ii = 0;
	bgc->bgv_buf = bgv_buf;
	//~ printf("%ld: [",b_start);
	for (ii=0; ii<BENG_FRAMES_PER_GROUP; ii++) {
		beng_frame_completion_t *bfc = &bgc->bfc[ii];
		// set all zeros
		memset(bfc,0,sizeof(beng_frame_completion_t));
		// then set B-counter value
		bfc->b = b_start+ii;
		//~ if (ii==0 || ii==(BENG_FRAMES_PER_BLOCK-1)) {
			//~ printf("%ld",bcmp->b);
			//~ if (ii==0) {
				//~ printf(" ...");
			//~ }
		//~ }
	}
	//~ printf("]\n");
}

int get_beng_group_index_offset(beng_group_completion_buffer_t *bgc_buf, int index_ref, vdif_in_packet_t *vdif_pkt) {
	int offset = -1;
	int count = BENG_GROUPS_IN_BUFFER;
	int64_t b = get_packet_b_count(&vdif_pkt->header);
	//~ printf("b=%ld\n",b);
	int64_t ll = bgc_buf->bgc[index_ref].bfc[0].b;
	int64_t lu = bgc_buf->bgc[index_ref].bfc[BENG_FRAMES_PER_GROUP-1].b;
	//~ printf("ll,lu=%ld,%ld\n",ll,lu);
	if (b < ll) {
		//~ printf("offset=%d\n",offset);
		return offset;
	}
	while (count-->0) {
		offset++;
		if (b >= ll && b <= lu) {
			//~ printf("offset=%d\n",offset);
			return offset;
		}
		// due to overlap the increment is one less than frames/block
		ll += BENG_FRAMES_PER_GROUP-1;
		lu += BENG_FRAMES_PER_GROUP-1;
		//~ printf("ll,lu=%ld,%ld\n",ll,lu);
	}
	//~ printf("offset=%d\n",offset);
	return offset;
}

int insert_vdif_in_beng_group_buffer(beng_group_completion_buffer_t *bgc_buf, int index_ref, int offset, vdif_in_packet_t *vdif_pkt) {
	int insert_count = 0;
	int ii = 0;
	int64_t b = get_packet_b_count(&vdif_pkt->header);
	//~ int64_t ll_min = (int64_t)0x3FFFFFFFFFFFFFFF;
	//~ int64_t lu_max = (int64_t)-1;
	//~ printf("b=%ld\n",b);
	for (ii=offset; ii<BENG_GROUPS_IN_BUFFER; ii++) {
		int g_idx = (ii+index_ref) % BENG_GROUPS_IN_BUFFER;
		beng_group_completion_t *this_bgc = &bgc_buf->bgc[g_idx];
		int64_t ll = this_bgc->bfc[0].b;
		int64_t lu = this_bgc->bfc[BENG_FRAMES_PER_GROUP-1].b;
		//~ printf("ll,lu=%ld,%ld\n",ll,lu);
		if (b >= ll && b <= lu) {// B-frame counter in range
			int b_idx = b - this_bgc->bfc[0].b;
			int c_idx = vdif_pkt->header.beng.c;
			int f_idx = vdif_pkt->header.beng.f;
			//~ printf("b_idx=%d, c_idx=%d, f_idx=%d\n",b_idx,c_idx,f_idx);
			if (this_bgc->bfc[b_idx].beng_frame_vdif_packet_count < VDIF_PER_BENG_FRAME && // cannot exceed maximum VDIF per B-frame
				this_bgc->beng_group_vdif_packet_count < (BENG_FRAMES_PER_GROUP*VDIF_PER_BENG_FRAME)) { // cannot exceed maximum VDIF per group of B-frames
				uint8_t *f_flags = (uint8_t *)&this_bgc->bfc[b_idx].cc[c_idx];
				// update the B-frame completion ...
				this_bgc->bfc[b_idx].beng_frame_vdif_packet_count++;
				//~ printf("..bframe_vdif_packet_count=%d\n",this_bgc->bfc[b_idx].beng_frame_vdif_packet_count);
				*f_flags |= (0x01 << f_idx);
				//~ printf("f_flags=%x\n",(int)*f_flags);
				// ... copy the VDIF packet to memory ...
				memcpy(this_bgc->bgv_buf+this_bgc->beng_group_vdif_packet_count,vdif_pkt,sizeof(vdif_in_packet_t));
				// and increment packet counter
				this_bgc->beng_group_vdif_packet_count++;
				//~ printf(".total_vdif_packet_count=%d\n",this_bgc->beng_group_vdif_packet_count);
				// increment insertion count
				insert_count++;
			}
		}
	}
	//~ printf("insert_count=%d\n",insert_count);
	return insert_count;
}

int check_beng_group_complete(beng_group_completion_buffer_t *bgc_buf,int index) {
	if (index < 0 || index >= BENG_GROUPS_IN_BUFFER) {
		return -1;
	}
	beng_group_completion_t *this_bgc = &bgc_buf->bgc[index];
	if (this_bgc->beng_group_vdif_packet_count == (BENG_FRAMES_PER_GROUP*VDIF_PER_BENG_FRAME)) {
		return 1;
	}
	return 0;
}
