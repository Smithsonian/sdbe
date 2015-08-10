#ifndef VDIF_IN_DATABUF_CUDA_H
#define VDIF_IN_DATABUF_CUDA_H

#include "vdif_in_databuf.h"

void set_ipc_mem_handle(cudaIpcMemHandle_t *ipc_mem_handle, beng_group_vdif_buffer_t *ptr_gpu);

int get_bgv_cpu_memory_cuda(beng_group_vdif_buffer_t **bgv_buf_cpu, int index);

int get_bgv_gpu_memory_cuda(beng_group_vdif_buffer_t **bgv_buf_gpu, int index);

// CUDA code to initiate data copy
int transfer_beng_group_to_gpu_cuda(vdif_in_databuf_t *bgc_buf, int index);

// CUDA code to check if data copy is complete
int check_transfer_complete_cuda(vdif_in_databuf_t *bgc_buf, int index);

#endif // VDIF_IN_DATABUF_CUDA_H
