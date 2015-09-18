/*
Inverse PFB following Richard Shaw's original python/LAPACK routine: https://github.com/jrs65/pfb-inverse
@author Katherine Rosenfeld
@date 8/2015

To compile:
  nvcc pfb_inverse_cusolver.cu -o pfb_inverse_cusolver.out -lcusolver -lcusparse -lcurand -lcufft -lcublas

Resources : http://devblogs.nvidia.com/parallelforall/parallel-direct-solvers-with-cusolver-batched-qr/
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <curand.h>
#include <cufft.h>

#define BENG_CHANNELS_ 16384
#define BENG_SNAPSHOTS 128
#define PI 3.14159265359

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(x));\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUSPARSE_CALL(x) do { if((x)!=CUSPARSE_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__host__ __device__ float hamming(int n, int m){
  return 0.54 - 0.46*cos(2.*PI*n/(m-1.));
}

// decimation kernel
__global__ void decimate(cufftComplex *in, cufftComplex *out, int M, int N){
  int tid = blockIdx.x*blockDim.x + threadIdx.x; 
  for (int i=tid; i<N; i+= gridDim.x*blockDim.x){
    if (i % M == 0) {
      out[i / M] = in[i];
    }
  }
}

// multiple kernel
__global__ void multiply(float *a, float b, int N){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i=tid; i<N; i+=gridDim.x*blockDim.x){
    a[i] *= b;
  }
}

// cross multiply kernel
__global__ void cross_multiply(cufftComplex *S_0x1, cufftComplex *X0, cufftComplex *X1, int N){
  // returns S_0x1 = X0 * conj(X1)
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = tid; i < N; i += blockDim.x*gridDim.x){
    S_0x1[i].x = X0[i].x*X1[i].x + X0[i].y*X1[i].y;
    S_0x1[i].y = X0[i].y*X1[i].x - X0[i].x*X1[i].y;
  }
}

// compute mean along column [m x n, row major format]
__global__ void col_mean(cufftComplex *in, int m, int n){
  int cid = blockIdx.x*blockDim.x + threadIdx.x;
  // stride along column id
  for (int i = cid; i < n; i += gridDim.x*blockDim.x){
    float avg_re = 0;
    float avg_im = 0;
    for (int j = 0 ; j < m; j++){
      avg_re += in[i + j*n].x;
      avg_im += in[i + j*n].y;
    }
      //in[i] = make_cuComplex(avg_re / m, avg_im / m);
      in[i].x = avg_re/m;
      in[i].y = avg_im/m;
  }
}

// apply window function
__global__ void window(float *in, float *out, int N){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i=tid; i<N; i+= gridDim.x*blockDim.x){
	out[i] = in[i]*hamming(i,N);
  }
}

float corr_FXt(float *d_x0, float *d_x1, int num_samples){
  int idx,window_size = 32768;
  cufftHandle plan,iplan;
  cublasHandle_t handle;
  //int batch = num_samples / window_size;
  int batch = (1 << (int) floor(log2((float)num_samples))) / window_size;
  cufftComplex *d_S,*d_X0, *d_X1;
  dim3 blocks(64,1,1),threads(256,1,1);
  float *d_s;
  float s0x0_max, s1x1_max, corr_coeff;

  printf("%s : batch = %d \n",__FUNCTION__, batch);

  // allocate device arrays
  CUDA_CALL( cudaMalloc((void **) &d_X0, (window_size/2+1)*batch*sizeof(cufftComplex)) );
  CUDA_CALL( cudaMalloc((void **) &d_X1, (window_size/2+1)*batch*sizeof(cufftComplex)) );
  CUDA_CALL( cudaMalloc((void **) &d_S,  (window_size/2+1)*batch*sizeof(cufftComplex)) );
  CUDA_CALL( cudaMalloc((void **) &d_s,  window_size*sizeof(float)) );

  // create FFT plans and cuBLAS handle
  CUFFT_CALL( cufftPlanMany(&plan, 1, &window_size, NULL,1,0,NULL,1,0,CUFFT_R2C,batch) );
  CUFFT_CALL( cufftPlanMany(&iplan, 1, &window_size, NULL,1,0,NULL,1,0,CUFFT_C2R,1) );
  CUBLAS_CALL( cublasCreate(&handle) );

  // execute R2C FFT
  CUFFT_CALL( cufftExecR2C(plan, d_x0, d_X0) );
  CUFFT_CALL( cufftExecR2C(plan, d_x1, d_X1) );

  // auto-corr X0, X0
  cross_multiply<<<blocks,threads>>>(d_S,d_X0,d_X0,batch*(window_size/2+1));
  col_mean<<<blocks,threads>>>(d_S,batch,window_size/2+1);
  CUFFT_CALL( cufftExecC2R(iplan, d_S, d_s) );
  CUBLAS_CALL( cublasIsamax(handle, window_size, d_s, 1, &idx) );
  CUDA_CALL( cudaMemcpy( &s0x0_max, d_s + (idx-1), 1*sizeof(float), cudaMemcpyDeviceToHost) );

  // auto-corr X1, X1
  cross_multiply<<<blocks,threads>>>(d_S,d_X1,d_X1,batch*(window_size/2+1));
  col_mean<<<blocks,threads>>>(d_S,batch,window_size/2+1);
  CUFFT_CALL( cufftExecC2R(iplan, d_S, d_s) );
  CUBLAS_CALL( cublasIsamax(handle, window_size, d_s, 1, &idx) );
  CUDA_CALL( cudaMemcpy( &s1x1_max, d_s + (idx-1), 1*sizeof(float), cudaMemcpyDeviceToHost) );

  // cross-corr X0, X1
  cross_multiply<<<blocks,threads>>>(d_S,d_X0,d_X1,batch*(window_size/2+1));
  col_mean<<<blocks,threads>>>(d_S,batch,window_size/2+1);
  CUFFT_CALL( cufftExecC2R(iplan, d_S, d_s) );
  CUBLAS_CALL( cublasIsamax(handle, window_size, d_s, 1, &idx) );
  CUDA_CALL( cudaMemcpy( &corr_coeff, d_s + (idx-1), 1*sizeof(float), cudaMemcpyDeviceToHost) );
  printf("corr coeff: %f %d \n",corr_coeff/sqrt(s1x1_max*s0x0_max), idx);


  // clean up
  CUFFT_CALL( cufftDestroy(plan) );
  CUFFT_CALL( cufftDestroy(iplan) );
  CUDA_CALL( cudaFree(d_X0) );
  CUDA_CALL( cudaFree(d_X1) );
  CUDA_CALL( cudaFree(d_S) );
  CUDA_CALL( cudaFree(d_s) );
  CUBLAS_CALL( cublasDestroy(handle) );
  return corr_coeff/sqrt(s1x1_max*s0x0_max); 
}


// generate pfb spectrum (doesn't actually do the polyphase bit...)
int pfb(float *d_t, int num_samples, int num_tap, int num_freq, cufftComplex *d_s){
  int lblock = 2 * (num_freq - 1);
  int nblock = num_samples / lblock - (num_tap - 1);
  float *d_tt;
  cufftComplex *d_ft;
  cufftHandle plan;

  // create FFT plan
  int batch = 1;
  int fft_size = lblock*num_tap;
  CUDA_CALL( cudaMalloc((void **) &d_ft, (fft_size/2+1)*sizeof(cufftComplex)) ); 
  CUDA_CALL( cudaMalloc((void **) &d_tt, fft_size*sizeof(cufftComplex)) ); 
  CUFFT_CALL( cufftPlanMany(&plan, 1, &fft_size,NULL,1,0,NULL,1,0,CUFFT_R2C,batch) );

  dim3 blocks(64,1,1);
  dim3 threads(512,1,1);

  // iterate over blocks (no batches yet)
  for (int i=0; i < nblock; i++){

	// window
	window<<<blocks,threads>>>(d_t + i*lblock, d_tt, fft_size);
	CUDA_CALL(cudaGetLastError());

	// execute rFFT
  	CUFFT_CALL( cufftExecR2C(plan, d_tt, d_ft) );

	// decimate
	decimate<<<blocks,threads>>>(d_ft,d_s+i*num_freq,num_tap,fft_size/2+1);
	CUDA_CALL(cudaGetLastError());
  }

  CUDA_CALL( cudaFree(d_ft) );
  CUDA_CALL( cudaFree(d_tt) );
  CUFFT_CALL( cufftDestroy(plan) );
  return 1;
}

/*
 d_s is complex PFB timestream [num_snapshots, num_freqs]
*/
int inverse_pfb(cufftComplex *d_s, int num_samples, int num_tap, int num_freq, float *d_rts){
  // pull out the number of blocks and their length
  int lblock = 2 * (num_freq - 1);
  int nblock = num_samples / lblock - (num_tap - 1);
  int ntsblock = nblock + num_tap - 1;

  cudaEvent_t tic,toc;
  float elapsedTime;
  cublasHandle_t cublasH;
  cusparseHandle_t cusparseH;
  cufftHandle plan;
  float *d_pts, *d_yh;
  const float alpha = 1., beta = 0.;

  cusolverSpHandle_t cusolverH = NULL;
  // GPU does batch QR
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrBandPPT = NULL;
  cusparseMatDescr_t descrBandP = NULL;
  cusolverStatus_t cusolver_status;
  cusparseStatus_t cusparse_status;
  cublasStatus_t cublas_status;

  // create CUDA events for internal timing
  cudaEventCreate(&tic);
  cudaEventCreate(&toc);

  // create cublas context
  CUBLAS_CALL( cublasCreate(&cublasH) );

  // create cusparse context
  CUSPARSE_CALL( cusparseCreate(&cusparseH) );

  // GPU does batch QR
  // batchsize = lblock
  // m = nblock
  // d_A is CSR format, d_csrValA is of size nnzA*batchSize = 
  // d_x is a matrix of size batchSize * m
  // d_b is a matrix of size batchSize * m
  int *csrRowPtrBandPPT,*csrColIndBandPPT, *csrRowPtrBandP, *csrColIndBandP;
  int *d_csrRowPtrBandPPT,*d_csrColIndBandPPT, *d_csrRowPtrBandP, *d_csrColIndBandP;
  float *csrValBandPPT, *csrValBandP;
  float *d_csrValBandPPT, *d_csrValBandP;
  int ind, u = num_tap - 1;
  size_t size_qr = 0;
  size_t size_internal = 0;
  void *buffer_qr = NULL; // working space for numerical factorization

  // set up banded P matrices
  int mBandP = lblock * ntsblock, nBandP = lblock * nblock;
  int nnzBandP = lblock * num_tap * nblock; 
  csrRowPtrBandP = (int *) malloc((mBandP + 1)*sizeof(int));
  csrColIndBandP = (int *) malloc(nnzBandP*sizeof(int));
  csrValBandP = (float *) malloc(nnzBandP*sizeof(float));

  ind = 0;
  for (int batchId = 0; batchId < lblock; batchId++){
    for (int j = 0; j < ntsblock; j++){
      csrRowPtrBandP[batchId*ntsblock + j] = ind;
      for (int i = 0; i < nblock; i++){
        if (((u+i-j) >= 0) && (i <= j)) {
          csrColIndBandP[ind] = batchId*nblock + i;
          csrValBandP[ind] = hamming(lblock*(j-i) + batchId, num_tap*lblock);
          ++ind;
        }
      }
    }
  }
  csrRowPtrBandP[mBandP] = nnzBandP;

  // set up banded P*P^T matrices
  int mBandPPT = nblock;
  int nnzBandPPT = 2 * (num_tap*nblock - (num_tap - 1) * num_tap / 2) - nblock;
  csrRowPtrBandPPT = (int *) malloc((mBandPPT + 1)*sizeof(int));
  csrColIndBandPPT = (int *) malloc(nnzBandPPT*sizeof(int));
  csrValBandPPT = (float *) malloc(lblock*nnzBandPPT*sizeof(float));

  for (int batchId = 0; batchId < lblock; batchId++){  
    ind = 0;
    // loop over rows
    for (int j=0; j < mBandPPT; j++){
      csrRowPtrBandPPT[j] = ind;
      // loop over columns
      for (int i=0; i < mBandPPT; i++){
        int dji = abs(j-i);
        if ((u-dji) >= 0) {
          float val = 0.;
          for (int k = 0; k < num_tap - dji; k ++) {
            val += hamming(lblock*(k+dji) + batchId, num_tap*lblock) * hamming(lblock*k + batchId, num_tap*lblock);
          }
          csrColIndBandPPT[ind] = i;
          csrValBandPPT[nnzBandPPT*batchId + ind] = val;
          ++ind;
        }
      }
    }
    csrRowPtrBandPPT[mBandPPT] = nnzBandPPT;
  }

#if 0
  // banded P
  bandP = (float*) malloc(lblock*num_tap*ntsblock*sizeof(float));
  for (int k=0; k<lblock; k++){
    for (int j=0; j<num_tap; j++){
      for (int i=0; i<ntsblock; i++){
        //band_P[k*ntap*ntsblock + i*ntap + j] = coeff_P[(num_tap-1-j)*lblock + k];
        bandP[k*num_tap*ntsblock + i*num_tap + j] = hamming((num_tap-1-j)*lblock + k,lblock * num_tap);
      }
    }
  }


  FILE *pFile;

  pFile = fopen("bandPPT_data.txt","w");
  for (int i=0; i<nnz; i++)  fprintf(pFile,"%e\n", csrValBandPPT[i]);
  fclose(pFile);
  pFile = fopen("bandPPT_indptr.txt","w");
  for (int i=0; i<m+1; i++)  fprintf(pFile,"%d\n", csrRowPtrBandPPT[i]);
  fclose(pFile);
  pFile = fopen("bandPPT_indices.txt","w");
  for (int i=0; i<nnz; i++)  fprintf(pFile,"%d\n", csrColIndBandPPT[i]);
  fclose(pFile);

  // this looks to match csr.py
#endif

  // copy banded P*P^T matrices to device
  //CUDA_CALL( cudaMalloc((void**)&d_bandP, ntsblock*lblock*num_tap*sizeof(float)) );
  //CUDA_CALL( cudaMemcpy(d_bandP, bandP, ntsblock*lblock*num_tap*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMalloc((void **) &d_yh, ntsblock*lblock*sizeof(float)) );
  CUDA_CALL( cudaMalloc((void**)&d_csrValBandPPT, sizeof(float)*nnzBandPPT*lblock) );
  CUDA_CALL( cudaMalloc((void**)&d_csrColIndBandPPT, sizeof(int)*nnzBandPPT) );
  CUDA_CALL( cudaMalloc((void**)&d_csrRowPtrBandPPT, sizeof(int)*(mBandPPT+1)) );
  CUDA_CALL( cudaMalloc((void**)&d_csrValBandP, sizeof(float)*nnzBandP) );
  CUDA_CALL( cudaMalloc((void**)&d_csrColIndBandP, sizeof(int)*nnzBandP) );
  CUDA_CALL( cudaMalloc((void**)&d_csrRowPtrBandP, sizeof(int)*(mBandP+1)) );
  CUDA_CALL( cudaMemcpy(d_csrValBandPPT, csrValBandPPT, sizeof(float)*nnzBandPPT*lblock, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrColIndBandPPT, csrColIndBandPPT, sizeof(int)*nnzBandPPT, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrRowPtrBandPPT, csrRowPtrBandPPT, sizeof(int)*(mBandPPT+1), cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrValBandP, csrValBandP, sizeof(float)*nnzBandP, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrColIndBandP, csrColIndBandP, sizeof(int)*nnzBandP, cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_csrRowPtrBandP, csrRowPtrBandP, sizeof(int)*(mBandP+1), cudaMemcpyHostToDevice) );

  // create cusolver handle, qr info, and matrix descriptor
  CUSOLVER_CALL( cusolverSpCreate(&cusolverH) );
  CUSPARSE_CALL( cusparseCreateMatDescr(&descrBandPPT) );
  CUSPARSE_CALL( cusparseCreateMatDescr(&descrBandP) );
  //CUSPARSE_CALL( cusparseSetMatType(descrBandPPT, CUSPARSE_MATRIX_TYPE_GENERAL) );
  //CUSPARSE_CALL( cusparseSetMatIndexBase(descrBandPPT, CUSPARSE_INDEX_BASE_ZERO) ); // base-0
  CUSOLVER_CALL( cusolverSpCreateCsrqrInfo(&info) );	// for batched execution

  // symbolic analysis
  cusolver_status = cusolverSpXcsrqrAnalysisBatched(
		cusolverH, mBandPPT, mBandPPT, nnzBandPPT,
		descrBandPPT, d_csrRowPtrBandPPT, d_csrColIndBandPPT,
		info);
  CUSOLVER_CALL( cusolver_status );

  // prepare working space
  cusolver_status = cusolverSpScsrqrBufferInfoBatched(
		cusolverH, mBandPPT, mBandPPT, nnzBandPPT,
		descrBandPPT, d_csrValBandPPT, d_csrRowPtrBandPPT, d_csrColIndBandPPT,
		lblock,
		info,
		&size_internal,
		&size_qr);
  CUSOLVER_CALL( cusolver_status );

  printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);      
  printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);      

  cudaEventRecord(tic);

  // generate pseudo timestream
  CUDA_CALL( cudaMalloc((void **) &d_pts, ntsblock*lblock*sizeof(float)) );
  CUFFT_CALL( cufftPlanMany(&plan, 1, &lblock, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, nblock) );
  CUFFT_CALL( cufftExecC2R(plan, d_s, d_yh) );

#if 0

  // transpose the nblock x lblock vectors to lblock x nblock
  // cufft assumes row major format, cublas assumes collumn major format
  // http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-geam
  cublas_status = cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
			nblock, lblock,
			&alpha, d_yh, lblock,
			&beta, NULL, nblock,
			d_pts, nblock);
  if (cublas_status != CUBLAS_STATUS_SUCCESS){
    printf("Error at %s:%s:%d\n",__FILE__,__FUNCTION__,__LINE__);
  }

  // multiple pseudo-timestream by 1./lblock (to rescale inverse FFT)
  dim3 blocks(64,1,1);
  dim3 threads(512,1,1);
  multiply<<<blocks,threads>>>(d_pts,1./lblock,lblock*nblock);

  // solve for intermediate vector
  CUDA_CALL( cudaMalloc((void**)&buffer_qr, size_qr) );
  cusolver_status = cusolverSpScsrqrsvBatched(
			cusolverH, mBandPPT, mBandPPT, nnzBandPPT,
			descrBandPPT, d_csrValBandPPT, d_csrRowPtrBandPPT, d_csrColIndBandPPT,
			d_pts, d_yh,
			lblock,
			info,
			buffer_qr);
  CUSOLVER_CALL( cusolver_status );

  // project back onto time stream
  // http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrmv
  cusparse_status = cusparseScsrmv(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
			mBandP, nBandP, nnzBandP, &alpha,
			descrBandP, d_csrValBandP, d_csrRowPtrBandP, d_csrColIndBandP,
			d_yh, &beta,
			d_pts);
  CUSPARSE_CALL( cusparse_status );

#if 0
  // project back onto time stream
  for (int i=0; i < lblock; i++){
    // project back onto time-stream
    // http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gbmv
    cublas_status = cublasSgbmv(cublasH,CUBLAS_OP_T,
		nblock, ntsblock, 0, num_tap-1,
		&alpha,d_bandP+i*num_tap*ntsblock,num_tap,
		d_yh+i*nblock, 1, 
		&beta, d_pts+i*ntsblock, 1
	);
     CUBLAS_CALL( cublas_status );
  }
#endif

  // now transpose lblock x ntsblock to ntsblock x lblock timeseries
  // but remember that cublas is column major 
  cublas_status = cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
			lblock, ntsblock,
			&alpha, d_pts, ntsblock,
			&beta, NULL, lblock,
			d_rts, lblock);
  CUBLAS_CALL( cublas_status );

#endif

  cudaEventRecord(toc);
  cudaEventSynchronize(toc);
  cudaEventElapsedTime(&elapsedTime,tic,toc);
  printf("inverse-pfb (gpu only): %f ms\n",elapsedTime);


  // clean up
  free(csrRowPtrBandPPT);
  free(csrColIndBandPPT);
  free(csrValBandPPT);
  //free(bandP);
  CUDA_CALL( cudaEventDestroy(tic) );
  CUDA_CALL( cudaEventDestroy(toc) );
  CUDA_CALL( cudaFree(buffer_qr) );
  CUDA_CALL( cudaFree(d_yh) );
  CUDA_CALL( cudaFree(d_pts) );
  //CUDA_CALL( cudaFree(d_bandP) );
  CUDA_CALL( cudaFree(d_csrValBandPPT) );
  CUDA_CALL( cudaFree(d_csrColIndBandPPT) );
  CUDA_CALL( cudaFree(d_csrRowPtrBandPPT) );
  CUDA_CALL( cudaFree(d_csrValBandP) );
  CUDA_CALL( cudaFree(d_csrColIndBandP) );
  CUDA_CALL( cudaFree(d_csrRowPtrBandP) );
  CUFFT_CALL( cufftDestroy(plan) );
  CUBLAS_CALL( cublasDestroy(cublasH) );
  CUSOLVER_CALL( cusolverSpDestroyCsrqrInfo(info) );
  //CUSOLVER_CALL( cusolverSpDestroy(cuSolverH) );
  CUSPARSE_CALL( cusparseDestroyMatDescr(descrBandPPT) );
  CUSPARSE_CALL( cusparseDestroyMatDescr(descrBandP) );
  CUSPARSE_CALL( cusparseDestroy(cusparseH) );
  return 1;
}


int main(int argc, char* argv[]){
  //int num_snapshots = 39;
  int num_snapshots = 32;
  int num_tap = 4, num_freq = BENG_CHANNELS_ + 1;
  float elapsedTime;
  float *d_ts, *d_rts;
  cufftComplex *d_s;
  cudaEvent_t tic, toc;
  curandGenerator_t gen;

  //int num_samples = 2*BENG_CHANNELS_*(BENG_SNAPSHOTS*num_beng_frames + num_tap - 1);
  int num_samples = 2*BENG_CHANNELS_*(num_snapshots + num_tap - 1);
  int lblock = 2 * (num_freq - 1);
  int nblock = num_samples / lblock - (num_tap - 1);

  printf("num_samples=%d\n",num_samples);
  printf("num_freqs=%d\n",num_freq);
  printf("lblock=%d\n",lblock);
  printf("nblock=%d\n",nblock);

  // create events
  CUDA_CALL( cudaEventCreate(&tic) );
  CUDA_CALL( cudaEventCreate(&toc) );

  // allocate device memory
  CUDA_CALL( cudaMalloc((void **) &d_ts, num_samples*sizeof(float)) );
  CUDA_CALL( cudaMalloc((void **) &d_s, nblock*num_freq*sizeof(cufftComplex)) ); 

  // generate data
  CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  CUDA_CALL( cudaEventRecord(tic) );
  CURAND_CALL(curandGenerateNormal(gen, d_ts, num_samples, 0., 1.) );
  CUDA_CALL( cudaEventRecord(toc) );
  CUDA_CALL( cudaEventSynchronize(toc) );
  CUDA_CALL( cudaEventElapsedTime(&elapsedTime,tic,toc) ); 
  fprintf(stdout, "generating %d random numbers took %f ms\n",num_samples,elapsedTime);

  // pfb
  CUDA_CALL( cudaEventRecord(tic) );
  pfb(d_ts, num_samples, num_tap, num_freq, d_s);
  CUDA_CALL( cudaEventRecord(toc) );
  CUDA_CALL( cudaEventSynchronize(toc) );
  CUDA_CALL( cudaEventElapsedTime(&elapsedTime,tic,toc) ); 
  fprintf(stdout, "pfb took %f ms\n",elapsedTime);

  // inverse pfb
  CUDA_CALL( cudaMalloc((void **) &d_rts, num_samples*sizeof(float)) );
  CUDA_CALL( cudaEventRecord(tic) );
  inverse_pfb(d_s, num_samples, num_tap, num_freq, d_rts);
  CUDA_CALL( cudaEventRecord(toc) );
  CUDA_CALL( cudaEventSynchronize(toc) );
  CUDA_CALL( cudaEventElapsedTime(&elapsedTime,tic,toc) ); 
  fprintf(stdout, "inverse-pfb took %f ms\n",elapsedTime);

  // compute the correlation coefficient here:
  CUDA_CALL( cudaEventRecord(tic) );
  float corr_coeff = corr_FXt(d_rts,d_ts, num_samples);
  CUDA_CALL( cudaEventRecord(toc) );
  CUDA_CALL( cudaEventSynchronize(toc) );
  CUDA_CALL( cudaEventElapsedTime(&elapsedTime,tic,toc) ); 
  fprintf(stdout, "FXcorr took %f ms\n",elapsedTime);

#if 0
  float *ts, *rts;
  // write time streams to file
  ts =  (float*) malloc(num_samples*sizeof(float));
  rts = (float*) malloc(num_samples*sizeof(float)); 
  CUDA_CALL( cudaMemcpy(ts, d_ts, num_samples*sizeof(float), cudaMemcpyDeviceToHost) );
  CUDA_CALL( cudaMemcpy(rts, d_rts, num_samples*sizeof(float), cudaMemcpyDeviceToHost) );

  FILE *pFile;
  pFile = fopen("ts.txt","w");
  for (int i=0; i < num_samples; i++){
    fprintf(pFile,"%e %e\n",ts[i], rts[i]);
  }
  fclose(pFile);

  free(ts);
  free(rts);
#endif

  // clean up
  CURAND_CALL( curandDestroyGenerator(gen) );
  CUDA_CALL( cudaEventDestroy(tic) );
  CUDA_CALL( cudaEventDestroy(toc) );
  CUDA_CALL( cudaFree(d_ts) );
  CUDA_CALL( cudaFree(d_s) );
  CUDA_CALL( cudaFree(d_rts) );
  fprintf(stdout,"done!\n");
}
