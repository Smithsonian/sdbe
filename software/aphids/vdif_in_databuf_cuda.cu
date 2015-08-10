#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "vdif_in_databuf.h"
#include "vdif_in_databuf_cuda.h"

#define NUM_GPU 1

// Allocate memory for, create, and return pointer to CUDA stream
// Arguments:
// ----------
//   <none>
// Return:
// -------
//   void pointer to the created CUDA stream, or NULL on failure
static void *create_copy_stream();
// Destroy the CUDA stream
// Arguments:
// ----------
//   copy_stream_ptr -- void * to memory where an instance of
//     cudaStream_t is stored.
// Return:
// -------
//   1 on success, -1 on failure
// Notes:
// ------
//   If successful, copy_stream_ptr will be set to NULL
static int destroy_copy_stream(void *copy_stream_ptr);

#define ERR_PREFIX_LEN 256
#define ERR_PREFIX_FMT "%s(%d): In function '%s':"
char err_prefix[ERR_PREFIX_LEN];

static void handle_cuda_error(cudaError_t err) {
	if (err == cudaSuccess) {
		return;
	}
	fprintf(stderr,"%s cudaError is %d [%s]\n",err_prefix,(int)err,cudaGetErrorString(err));
	abort();
}

void set_ipc_mem_handle(cudaIpcMemHandle_t *ipc_mem_handle, beng_group_vdif_buffer_t *ptr_gpu) {
	cudaIpcGetMemHandle(ipc_mem_handle,ptr_gpu);
}

int get_bgv_cpu_memory_cuda(beng_group_vdif_buffer_t **bgv_buf_cpu, int index) {
	// NOTE: Device selection may not be necessary
	int device_id;
	// select device - same device for stream creation and async copy
	device_id = index % NUM_GPU;
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaSetDevice(device_id));
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaMallocHost((void **)bgv_buf_cpu, sizeof(beng_group_vdif_buffer_t)));
	return 1;
}

int get_bgv_gpu_memory_cuda(beng_group_vdif_buffer_t **bgv_buf_gpu, int index) {
	int device_id;
	// select device
	device_id = index % NUM_GPU;
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaSetDevice(device_id));
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaMalloc((void **)bgv_buf_gpu, sizeof(beng_group_vdif_buffer_t)));
	return 1;
}

int transfer_beng_group_to_gpu_cuda(vdif_in_databuf_t *bgc_buf, int index) {
	int device_id;
	// select device - same device for stream creation and async copy
	device_id = index % NUM_GPU;
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaSetDevice(device_id));
	// create separate CUDA stream for this copy
	cudaStream_t *cst;
	cst = create_copy_stream();
	bgc_buf->bgc[index].memcpy_stream = (void *)cst;
	bgc_buf->bgc[index].gpu_id = device_id;
	// start the copy
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaMemcpyAsync(bgc_buf->bgc[index].bgv_buf_gpu, bgc_buf->bgc[index].bgv_buf_cpu, sizeof(beng_group_vdif_buffer_t), cudaMemcpyHostToDevice, *(cudaStream_t *)bgc_buf->bgc[index].memcpy_stream));
	return 1;
}

int check_transfer_complete_cuda(vdif_in_databuf_t *bgc_buf, int index) {
	int device_id;
	cudaError_t result;
	// select device -- same device for stream query and possible destruction
	device_id = index % NUM_GPU;
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaSetDevice(device_id));
	// query the copy stream
	result = cudaStreamQuery(*(cudaStream_t *)bgc_buf->bgc[index].memcpy_stream);
	if (result == cudaErrorNotReady) {
		return 0;
	} else if (result != cudaSuccess) {
		snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
		handle_cuda_error(result);
	}
	// if we're here, then copy complete; destroy stream and return 1
	destroy_copy_stream(bgc_buf->bgc[index].memcpy_stream);
	return 1;
}

static void *create_copy_stream() {
	cudaStream_t *this_stream_ptr;
	this_stream_ptr = (cudaStream_t *)malloc(sizeof(cudaStream_t));
	if (this_stream_ptr == NULL) {
		fprintf(stderr,"%s(%d): In function '%s': NULL pointer\n",__FILE__,__LINE__,__FUNCTION__);
		abort();
	}
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaStreamCreate(this_stream_ptr));
	return (void *)this_stream_ptr;
}

static int destroy_copy_stream(void *copy_stream_ptr) {
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaStreamDestroy(*(cudaStream_t *)copy_stream_ptr));
	free(copy_stream_ptr);
	copy_stream_ptr = NULL;
	return 1;
}
