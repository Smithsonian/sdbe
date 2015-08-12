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

void print_quantized_storage(quantized_storage_t *qs, const char *tag) {
	printf("%s[QS: vdg_buf_cpu=%p, vdg_buf_gpu=%p, bit_depth=%d, N_32bit_words_per_chan=%d, ipc_mem_handle=?, gpu_id=%d, memcpy_stream=%p]\n",
			tag,qs->vdg_buf_cpu,qs->vdg_buf_gpu,qs->bit_depth,
			qs->N_32bit_words_per_chan,qs->gpu_id,qs->memcpy_stream);
}
