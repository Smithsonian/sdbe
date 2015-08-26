#include <stdlib.h>

#include "hashpipe.h"
#include "hashpipe_databuf.h"

#include "vdif_out_databuf.h"

#include "vdif_out_databuf_cuda.h"

hashpipe_databuf_t *vdif_out_databuf_create(int instance_id, int databuf_id)
{
  size_t header_size = sizeof(hashpipe_databuf_t);
  return hashpipe_databuf_create(instance_id, databuf_id, header_size, sizeof(quantized_storage_t), VDIF_OUT_BUFFER_SIZE);
}

int get_vdg_cpu_memory(vdif_out_data_group_t **vdg_buf_cpu, int index) {
	return get_vdg_cpu_memory_cuda(vdg_buf_cpu, index);
	//~ *bgv_buf_gpu = (beng_group_vdif_buffer_t *)malloc(sizeof(beng_group_vdif_buffer_t));
	//~ return 1;
}

int get_vpg_cpu_memory(vdif_out_packet_group_t **vpg_buf_cpu, int index) {
	*vpg_buf_cpu = (vdif_out_packet_group_t *)malloc(sizeof(vdif_out_packet_group_t));
	return vpg_buf_cpu == (void *)-1 ? -1 : 1;
}

int transfer_vdif_group_to_cpu(vdif_out_databuf_t *qs_buf, int index) {
	return transfer_vdif_group_to_cpu_cuda(qs_buf, index);
}

int check_transfer_vdif_group_to_cpu_complete(vdif_out_databuf_t *qs_buf, int index) {
	return check_transfer_vdif_group_to_cpu_complete_cuda(qs_buf, index);
}

void init_vdif_out(vdif_out_packet_group_t **vpg_buf_cpu, int index) {
	int ii,jj;
	
	for (ii=0; ii<VDIF_OUT_PKTS_PER_BLOCK; ii++) {
		for (jj=0; jj<VDIF_CHAN; jj++) {
			vdif_out_header_t *hdr = &vpg_buf_cpu[index]->chan[jj].packets[ii].header;
			// w0
			hdr->w0.secs_inre = 0; // needs updating for each packet
			hdr->w0.legacy = 0;
			hdr->w0.invalid = 0;
			// w1
			hdr->w1.df_num_insec = 0; // needs updating for each packet
			hdr->w1.ref_epoch = 0; // needs updating on first packet
			hdr->w1.UA = 0;
			// w2
			hdr->w2.df_len = sizeof(vdif_out_packet_t)/8;
			hdr->w2.num_channels = 0; // 2**num_channels(0) = 1 channel
			hdr->w2.ver = 0; // TODO: check correct value
			// w3
			hdr->w3.stationID = (((uint16_t)'S')&0x00FF<<8) | (((uint16_t)'m')&0x00FF); // TODO: check correct order
			hdr->w3.threadID = jj;
			hdr->w3.bps = 2; // needs updating
			hdr->w3.dt = 0; // real data
			// w4
			hdr->w4.edv5 = 0x02; // TODO: check correct value
			hdr->w4.eud5 = 0; // TODO: check correct value
			// w5
			hdr->w5.PICstatus = 0; // TODO: check correct value
			//
			hdr->edh_psn = 0; // needs updating for each packet
		}
	}
}

	//~ struct word0_out {
		//~ uint32_t secs_inre:30;
		//~ uint32_t legacy:1;
		//~ uint32_t invalid:1;
	//~ } w0;
	//~ struct word1_out {
		//~ uint32_t df_num_insec:24;
		//~ uint32_t ref_epoch:6;
		//~ uint32_t UA:2;
	//~ } w1;
	//~ struct word2_out {
		//~ uint32_t df_len:24;
		//~ uint32_t num_channels:5;
		//~ uint32_t ver:3;
	//~ } w2;
	//~ struct word3_out {
		//~ uint32_t stationID:16;
		//~ uint32_t threadID:10;
		//~ uint32_t bps:5;
		//~ uint32_t dt:1;
	//~ } w3;
	//~ struct word4_out {
		//~ uint32_t eud5:24;
		//~ uint32_t edv5:8;
	//~ } w4;
	//~ struct word5_out {
		//~ uint32_t PICstatus;
	//~ } w5;
	//~ uint64_t edh_psn;

void print_quantized_storage(quantized_storage_t *qs, const char *tag) {
	printf("%s[QS: vdg_buf_cpu=%p, vdg_buf_gpu=%p, bit_depth=%d, N_32bit_words_per_chan=%d, ipc_mem_handle=?, gpu_id=%d, memcpy_stream=%p]\n",
			tag,qs->vdg_buf_cpu,qs->vdg_buf_gpu,qs->bit_depth,
			qs->N_32bit_words_per_chan,qs->gpu_id,qs->memcpy_stream);
}
