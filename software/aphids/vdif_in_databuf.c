#include <stdlib.h>
#include <string.h>

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"
#include "vdif_in_databuf_cuda.h"


static void print_beng_frame_completion(beng_frame_completion_t *bfc, const char *tag);
static void print_channel_completion(channel_completion_t *cc, const char *tag);
static void print_beng_over_vdif_header(vdif_in_header_t *vdif_pkt_hdr, const char *tag);

#ifndef STANDALONE_TEST
hashpipe_databuf_t *vdif_in_databuf_create(int instance_id, int databuf_id)
{
  //~ fprintf(stderr,"%s:%d: Creating databuffer %d,%d...\n",__FILE__,__LINE__,instance_id,databuf_id);
  size_t header_size = sizeof(hashpipe_databuf_t);
  hashpipe_databuf_t *d = hashpipe_databuf_create(instance_id, databuf_id, header_size, sizeof(beng_group_completion_t), BENG_GROUPS_IN_BUFFER);
  //~ fprintf(stderr,"%s:%d: %p\n",__FILE__,__LINE__,d);
  return d;
}
#endif // STANDALONE_TEST

int64_t get_packet_b_count(vdif_in_header_t *vdif_pkt_hdr) {
	int64_t b = 0;
	b |= ((int64_t)(vdif_pkt_hdr->beng.b_upper)&(int64_t)0x00000000FFFFFFFF) << 8;
	b |= (int64_t)(vdif_pkt_hdr->beng.b_lower)&(int64_t)0x00000000000000FF;
	return b;
}

void init_beng_group(beng_group_completion_t *bgc, beng_group_vdif_buffer_t *bgv_buf_cpu, beng_group_vdif_buffer_t *bgv_buf_gpu, int64_t b_start) {
	int ii = 0;
	if (bgc->bgv_buf_cpu != bgv_buf_cpu || bgc->bgv_buf_gpu != bgv_buf_gpu) {
		bgc->bgv_buf_cpu = bgv_buf_cpu;
		bgc->bgv_buf_gpu = bgv_buf_gpu;
	}
	bgc->beng_group_vdif_packet_count = 0;
	for (ii=0; ii<BENG_FRAMES_PER_GROUP; ii++) {
		beng_frame_completion_t *bfc = &bgc->bfc[ii];
		// set all zeros
		memset(bfc,0,sizeof(beng_frame_completion_t));
		// then set B-counter value
		bfc->b = b_start+ii;
	}
	#ifdef STANDALONE_TEST
	//~ print_beng_group_completion(bgc,"init_beng_group: ");
	#endif
}

int get_beng_group_index_offset(vdif_in_databuf_t *bgc_buf, int index_ref, vdif_in_packet_t *vdif_pkt) {
	int offset = -1;
	int count = BENG_GROUPS_IN_BUFFER;
	int64_t b = get_packet_b_count(&vdif_pkt->header);
	int64_t ll = bgc_buf->bgc[index_ref].bfc[0].b;
	int64_t lu = bgc_buf->bgc[index_ref].bfc[BENG_FRAMES_PER_GROUP-1].b;
	if (b < ll) {
		return offset;
	}
	while (count-->0) {
		offset++;
		if (b >= ll && b <= lu) {
			return offset;
		}
		// due to overlap the increment is one less than frames/block
		ll += BENG_FRAMES_PER_GROUP-1;
		lu += BENG_FRAMES_PER_GROUP-1;
	}
	fprintf(stdout,"%s:%s(%d): b=%lu is outside search range [%lu,%lu]\n",__FILE__,__FUNCTION__,__LINE__,b,ll,lu);
	print_beng_over_vdif_header(&vdif_pkt->header, "BENG OFFSET TOO LARGE:");
	return offset;
}

int insert_vdif_in_beng_group_buffer(vdif_in_databuf_t *bgc_buf, int index_ref, int offset, vdif_in_packet_t *vdif_pkt) {
	int insert_count = 0;
	int ii = 0;
	int64_t b = get_packet_b_count(&vdif_pkt->header);
	for (ii=offset; ii<BENG_GROUPS_IN_BUFFER; ii++) {
		int g_idx = (ii+index_ref) % BENG_GROUPS_IN_BUFFER;
		beng_group_completion_t *this_bgc = &bgc_buf->bgc[g_idx];
		int64_t ll = this_bgc->bfc[0].b;
		int64_t lu = this_bgc->bfc[BENG_FRAMES_PER_GROUP-1].b;
		if (b >= ll && b <= lu) {// B-frame counter in range
			int b_idx = b - this_bgc->bfc[0].b;
			int c_idx = vdif_pkt->header.beng.c;
			int f_idx = vdif_pkt->header.beng.f;
			if (this_bgc->bfc[b_idx].beng_frame_vdif_packet_count < VDIF_PER_BENG_FRAME && // cannot exceed maximum VDIF per B-frame
				this_bgc->beng_group_vdif_packet_count < (BENG_FRAMES_PER_GROUP*VDIF_PER_BENG_FRAME)) { // cannot exceed maximum VDIF per group of B-frames
				uint8_t *f_flags = (uint8_t *)&this_bgc->bfc[b_idx].cc[c_idx];
				// update the B-frame completion ...
				this_bgc->bfc[b_idx].beng_frame_vdif_packet_count++;
				*f_flags |= (0x01 << f_idx);
				// ... copy the VDIF packet to memory ...
				memcpy((void *)(this_bgc->bgv_buf_cpu)+this_bgc->beng_group_vdif_packet_count*sizeof(vdif_in_packet_t),vdif_pkt,sizeof(vdif_in_packet_t));
				// and increment packet counter
				this_bgc->beng_group_vdif_packet_count++;
				// increment insertion count
				insert_count++;
			}
		}
	}
	//~ printf("insert_count=%d\n",insert_count);
	return insert_count;
}

int check_beng_group_complete(vdif_in_databuf_t *bgc_buf,int index) {
	if (index < 0 || index >= BENG_GROUPS_IN_BUFFER) {
		return -1;
	}
	beng_group_completion_t *this_bgc = &bgc_buf->bgc[index];
	if (this_bgc->beng_group_vdif_packet_count == (BENG_FRAMES_PER_GROUP*VDIF_PER_BENG_FRAME)) {
		return 1;
	}
	return 0;
}

int get_bgv_cpu_memory(beng_group_vdif_buffer_t **bgv_buf_cpu, int index) {
	return get_bgv_cpu_memory_cuda(bgv_buf_cpu, index);
	//~ *bgv_buf_cpu = (beng_group_vdif_buffer_t *)malloc(sizeof(beng_group_vdif_buffer_t));
	//~ return 1;
}

int get_bgv_gpu_memory(beng_group_vdif_buffer_t **bgv_buf_gpu, int index) {
	return get_bgv_gpu_memory_cuda(bgv_buf_gpu, index);
	//~ *bgv_buf_gpu = (beng_group_vdif_buffer_t *)malloc(sizeof(beng_group_vdif_buffer_t));
	//~ return 1;
}

int transfer_beng_group_to_gpu(vdif_in_databuf_t *bgc_buf, int index) {
	int rv;
	print_beng_group_completion(&bgc_buf->bgc[index], "");
	return transfer_beng_group_to_gpu_cuda(bgc_buf, index);
	//~ return 1;
}

int check_transfer_beng_group_to_gpu_complete(vdif_in_databuf_t *bgc_buf, int index) {
	return check_transfer_beng_group_to_gpu_complete_cuda(bgc_buf, index);
	//~ return 1;
}


// Print human-readable representation of B-engine group completion
void print_beng_group_completion(beng_group_completion_t *bgc, const char *tag) {
	printf("%s{B-engine group: beng_group_vdif_packet_count=%d, bgv_buf_cpu=%p, bgv_buf_gpu=%p, .bfc[0..39].b = %ld .. %ld\n",
			tag,bgc->beng_group_vdif_packet_count,bgc->bgv_buf_cpu,
			bgc->bgv_buf_gpu, bgc->bfc[0].b, bgc->bfc[BENG_FRAMES_PER_GROUP-1].b);
	//~ if (bgc->beng_group_vdif_packet_count < VDIF_PER_BENG_FRAME*BENG_FRAMES_PER_GROUP) {
		//~ int new_tag_len = strlen(tag) + 2;
		//~ char new_tag[new_tag_len];
		//~ snprintf(new_tag,new_tag_len,"%s  ",tag);
		//~ int ii = 0;
		//~ for (ii=0; ii<BENG_FRAMES_PER_GROUP; ii++) {
			//~ print_beng_frame_completion(bgc->bfc + ii, new_tag);
		//~ }
	//~ }
	printf("%s}\n",tag);
}

// Print human-readable representation of B-engine frame completion
static void print_beng_frame_completion(beng_frame_completion_t *bfc, const char *tag) {
	printf("%s[b=%ld; beng_frame_vdif_packet_count=%d",tag,bfc->b,bfc->beng_frame_vdif_packet_count);
	if (bfc->beng_frame_vdif_packet_count < VDIF_PER_BENG_FRAME) {
		printf("\n%s  Incomplete channels:",tag);
		int ii=0;
		for (ii=0; ii<CHAN_PER_BENG; ii++) {
			channel_completion_t *this_cc = bfc->cc + ii;
			int *this_cc_proxy = (int *)this_cc;
			if (*this_cc_proxy != 0xFF) {
				printf("\n%s    C-%03d: ",tag,ii);
				print_channel_completion(bfc->cc+ii,"");
			}
		}
		printf("\n%s",tag);
	}
	printf("]\n");
}

// Print human-readable representation of channel completion
static void print_channel_completion(channel_completion_t *cc, const char *tag) {
	printf("%s",tag);
	int ii=0;
	for (ii=0; ii<8; ii++) {
		int *cc_proxy = (int *)cc;
		printf("F-%d=%d ",ii,*cc_proxy>>ii & 0x01);
	}
}

static void print_beng_over_vdif_header(vdif_in_header_t *vdif_pkt_hdr, const char *tag) {
	fprintf(stdout,
			"%sw0.{secs_inre=%d, legacy=%d, invalid=%d}\n"
			"%sw1.{df_num_insec=%d, ref_epoch=%d, UA=%d}\n"
			"%sw2.{df_len=%d, num_channels=%d, ver=%d}\n"
			"%sw3.{stationID=%d, threadID=%d, bps=%d, dt=%d}\n"
			"%sbeng.{b_upper=%d, c=%d, z=%d, f=%d, b_lower=%d}\n"
			"%sedh_psn=%lu\n",
			tag,vdif_pkt_hdr->w0.secs_inre,vdif_pkt_hdr->w0.legacy,vdif_pkt_hdr->w0.invalid,
			tag,vdif_pkt_hdr->w1.df_num_insec,vdif_pkt_hdr->w1.ref_epoch,vdif_pkt_hdr->w1.UA,
			tag,vdif_pkt_hdr->w2.df_len,vdif_pkt_hdr->w2.num_channels,vdif_pkt_hdr->w2.ver,
			tag,vdif_pkt_hdr->w3.stationID,vdif_pkt_hdr->w3.threadID,vdif_pkt_hdr->w3.bps,vdif_pkt_hdr->w3.dt,
			tag,vdif_pkt_hdr->beng.b_upper,vdif_pkt_hdr->beng.c,vdif_pkt_hdr->beng.z,vdif_pkt_hdr->beng.f,vdif_pkt_hdr->beng.b_lower,
			tag,vdif_pkt_hdr->edh_psn);
}

//~ #ifdef STANDALONE_TEST
//~ #include <stdlib.h>
//~ int main(int argc, const char **argv) {
	//~ beng_group_completion_t bgc;
	//~ beng_group_vdif_buffer_t *bgv_buf_cpu;
	//~ beng_group_vdif_buffer_t *bgv_buf_gpu;
	//~ int64_t b_start = 371626;
	//~ 
	//~ // initialize VDIF buffers
	//~ bgv_buf_cpu = (beng_group_vdif_buffer_t *)malloc(sizeof(beng_group_vdif_buffer_t));
	//~ bgv_buf_gpu = (beng_group_vdif_buffer_t *)malloc(sizeof(beng_group_vdif_buffer_t));
	//~ 
	//~ // initialize completion
	//~ init_beng_group(&bgc, bgv_buf_cpu, bgv_buf_gpu, b_start);
	//~ 
	//~ // clean-up
	//~ free(bgv_buf_cpu);
	//~ free(bgv_buf_gpu);
	//~ 
	//~ return 0;
//~ }
//~ #endif // STANDALONE_TEST
