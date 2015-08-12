#include <stdio.h>
#include <stdlib.h>

#include "vdif_out_databuf.h"

#include "vdif_out_databuf_cuda.h"

#define NUM_GPU 1

#define ERR_PREFIX_LEN 256
#define ERR_PREFIX_FMT "%s(%d): In function '%s':"
char err_prefix[ERR_PREFIX_LEN];

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

static int get_vdg_gpu_memory_cuda(vdif_out_data_group_t **vdg_buf_gpu, cudaIpcMemHandle_t ipc_mem_handle);

static void handle_cuda_error(cudaError_t err) {
	cudaGetLastError();
	if (err == cudaSuccess) {
		return;
	}
	fprintf(stderr,"%s cudaError is %d [%s]\n",err_prefix,(int)err,cudaGetErrorString(err));
	abort();
}

int get_vdg_cpu_memory_cuda(vdif_out_data_group_t **vdg_buf_cpu, int index) {
	// NOTE: Device selection may not be necessary
	int device_id;
	// select device - same device for stream creation and async copy
	device_id = index % NUM_GPU;
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaSetDevice(device_id));
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaMallocHost((void **)vdg_buf_cpu, sizeof(vdif_out_data_group_t)));
	return 1;
}

static int get_vdg_gpu_memory_cuda(vdif_out_data_group_t **vdg_buf_gpu, cudaIpcMemHandle_t ipc_mem_handle) {
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	//~ handle_cuda_error(cudaIpcOpenMemHandle((void **)vdg_buf_gpu, ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
	handle_cuda_error(cudaMalloc((void **)vdg_buf_gpu, sizeof(vdif_out_data_group_t)));
	return 1;
}

int transfer_vdif_group_to_cpu_cuda(vdif_out_databuf_t *qs_buf, int index) {
	int device_id;
	// select device - same device for stream creation and async copy
	device_id = qs_buf->blocks[index].gpu_id;
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaSetDevice(device_id));
	// get reference to shared memory on GPU
	get_vdg_gpu_memory_cuda(&qs_buf->blocks[index].vdg_buf_gpu, qs_buf->blocks[index].ipc_mem_handle);
	// create separate CUDA stream for this copy
	cudaStream_t *cst;
	cst = create_copy_stream();
	qs_buf->blocks[index].memcpy_stream = (void *)cst;
	// start the copy
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaMemcpyAsync(qs_buf->blocks[index].vdg_buf_cpu, qs_buf->blocks[index].vdg_buf_gpu, sizeof(vdif_out_data_group_t), cudaMemcpyDeviceToHost, *(cudaStream_t *)qs_buf->blocks[index].memcpy_stream));
	return 1;
}

int check_transfer_vdif_group_to_cpu_complete_cuda(vdif_out_databuf_t *qs_buf, int index) {
	int device_id;
	//~ cudaError_t result;
	// select device -- same device for stream query and possible destruction
	device_id = qs_buf->blocks[index].gpu_id;
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaSetDevice(device_id));
	// query the copy stream
	//~ result = cudaStreamQuery(*(cudaStream_t *)qs_buf->blocks[index].memcpy_stream);
	snprintf(err_prefix,ERR_PREFIX_LEN,ERR_PREFIX_FMT,__FILE__,__LINE__,__FUNCTION__);
	handle_cuda_error(cudaStreamSynchronize(*(cudaStream_t *)qs_buf->blocks[index].memcpy_stream));
	// if we're here, close the shared memory reference ...
	//~ cudaIpcCloseMemHandle((void *)qs_buf->blocks[index].vdg_buf_gpu);
	handle_cuda_error(cudaFree((void *)qs_buf->blocks[index].vdg_buf_gpu));
	// ... and then copy complete; destroy stream and return 1
	destroy_copy_stream(qs_buf->blocks[index].memcpy_stream);
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

