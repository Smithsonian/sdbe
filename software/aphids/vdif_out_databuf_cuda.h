#ifndef VDIF_OUT_DATABUF_CUDA_H
#define VDIF_OUT_DATABUF_CUDA_H

#include "vdif_out_databuf.h"

int get_vdg_cpu_memory_cuda(vdif_out_data_group_t **vdg_buf_cpu, int index);

int transfer_vdif_group_to_cpu_cuda(vdif_out_databuf_t *qs_buf, int index);

int check_transfer_vdif_group_to_cpu_complete_cuda(vdif_out_databuf_t *qs_buf, int index);

#endif // VDIF_OUT_DATABUF_CUDA_H
