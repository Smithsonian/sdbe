#include <stdlib.h>
#include <string.h>

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_in_databuf.h"
#include "vdif_in_databuf_cuda.h"

#include "sample_rates.h"

// *********************************************************************
// The below two defintions are used to ensure alignment of the output
// data with real time, based on the number of input VDIF packets 
// skipped (due to an incomplete B-engine frame). The offset of the 
// input data with respect to real time can be calculated from the 
// following formula:
//
//   TIME_OFFSET_IN_BENG_FFT_WINDOWS = MAGIC_OFFSET_IN_BENG_FFT_WINDOWS - 128*(VDIF_PER_BENG_FRAME-N_SKIPPED_VDIF_PACKETS)/VDIF_PER_BENG_FRAME
//
// The value of MAGIC_OFFSET_IN_BENG_FFT_WINDOWS was empirically derived
// using cross-correlation between SDBE and single-dish reference data 
// for a couple of scans.
//
// The value of TIME_OFFSET_IN_BENG_FFT_WINDOWS is the number of 
// B-engine FFT windows that need to be skipped at the start of the 
// valid input data to obtain time alignment with the reference single 
// dish data, assuming the value is positive; if it is negative, it 
// means additional (and unavailable) data needs to be added before the 
// start of valid output data.
//
// TIME_OFFSET_IN_BENG_FFT_WINDOWS needs to be converted to a unit of 
// measure that can be accommodated in the output VDIF timestamps. The
// following formula converts a measurement in B-engine FFT windows to 
// a measurement in VDIF out packets:
//
//   TIME_OFFSET_IN_VDIF_OUT = MAGIC_BENG_FFT_WINDOW_IN_VDIF_OUT*TIME_OFFSET_IN_BENG_FFT_WINDOWS
//
// The value of MAGIC_BENG_FFT_WINDOW_IN_VDIF_OUT is based on the 
// following parameters:
//   * VDIF out packet length = 8us (1/VDIF_OUT_FRAMES_PER_SECOND)
//   * SWARM (effective) rate = 2496Msps
//   * Number of samples per SWARM FFT window = 32768
//~ #define MAGIC_OFFSET_IN_BENG_FFT_WINDOWS (52) <<--- definition moved to sample_rates.h
#define MAGIC_BENG_FFT_WINDOW_IN_VDIF_OUT ((float)32768/SWARM_RATE*VDIF_OUT_FRAMES_PER_SECOND)
// *********************************************************************

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
	if (vdif_pkt_hdr->w0.invalid) {
		return -1;
	}
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
	bgc->vdif_header_template.w0.invalid = 1;
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
	if (b < ll || vdif_pkt->header.w0.invalid) {
		if (vdif_pkt->header.w0.invalid) {
			return vidErrorPacketInvalid;
		} else {
			return vidErrorPacketBeforeStartTime;
		}
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
	#define BENG_OFFSET_TOO_LARGE_TO_CARE (BENG_FRAMES_PER_GROUP*1)
	if (b > lu + BENG_OFFSET_TOO_LARGE_TO_CARE) {
		return vidErrorPacketTooFarAheadToCare;
	}
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

void fill_vdif_header_template(vdif_in_header_t *vdif_hdr_copy, vdif_in_packet_t *vdif_pkt_ref, int n_skipped) {
	int offset_beng_fft_windows = 0;
	int offset_vdif_out_packets = 0;
	offset_beng_fft_windows = MAGIC_OFFSET_IN_BENG_FFT_WINDOWS - 128*(VDIF_PER_BENG_FRAME-n_skipped)/VDIF_PER_BENG_FRAME;
	offset_vdif_out_packets = (int)(MAGIC_BENG_FFT_WINDOW_IN_VDIF_OUT*offset_beng_fft_windows);
	fprintf(stdout,"%s:%s(%d): n_skipped = %d, offset_beng_fft_windows = %d, offset_vdif_out_packets = %d\n",__FILE__,__FUNCTION__,__LINE__,n_skipped,offset_beng_fft_windows,offset_vdif_out_packets);
	// do basic copy
	memcpy(vdif_hdr_copy, vdif_pkt_ref, sizeof(vdif_in_header_t));
	// then set timestamp information that should change
	if (offset_vdif_out_packets < 0) {
		vdif_hdr_copy->w0.secs_inre--;
		vdif_hdr_copy->w1.df_num_insec = 125000+offset_vdif_out_packets;
	} else {
		vdif_hdr_copy->w1.df_num_insec = offset_vdif_out_packets;
	}
	print_beng_over_vdif_header(vdif_hdr_copy,"TEMPLATE:");
	// the rest of the header should be updated as needed at the output stage
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
