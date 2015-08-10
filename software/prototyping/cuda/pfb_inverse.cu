/*
Inverse PFB following Richard Shaw's original python/LAPACK routine: https://github.com/jrs65/pfb-inverse
Beware: This implementation runs ~4x slower than the python version on hamster!
@author Katherine Rosenfeld
@date 8/2015

To compile:
  nvcc pfb_inverse.cu -o pfb_inverse.out -lcublas -lcurand -lcufft -llapack
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cufft.h>


#define BENG_CHANNELS_ 16384
#define BENG_SNAPSHOTS 128
#define PI 3.14159265359

extern "C" {
 void dpbtrf_(char* uplo, int *n, int* kd, double* ab, int* ldab, int* info);
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(x));\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
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

// apply window function
__global__ void window(float *in, float *out, int N){
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i=tid; i<N; i+= gridDim.x*blockDim.x){
	out[i] = in[i]*hamming(i,N);
  }
}


int PPT(int nblock, int lblock, int ntap, float *d_uPPT, float *d_band_P){
// http://www.physics.orst.edu/~rubin/nacphy/lapack/routines/spbtrf.html
  double *ab, *coeff_P, *coeff_PPT;
  float  *uPPT, *band_P;
  char uplo = 'U';	// store upper triangle
  int n = nblock, kd = ntap-1, ldab = ntap, info;
  int ntsblock = nblock + ntap - 1;

  // allocate memory
  coeff_P = (double *) malloc(ntap*lblock*sizeof(double));
  coeff_PPT = (double *) malloc(lblock*ntap*sizeof(double));
  ab = (double*) malloc(ntap*nblock*sizeof(double));
  uPPT = (float*) malloc(lblock*ntap*nblock*sizeof(float));
  band_P = (float*) malloc(lblock*ntap*ntsblock*sizeof(float));

  // generate window function
  for (int i=0; i<ntap*lblock; i++){
	coeff_P[i] = hamming(i,ntap*lblock);
  }

  for (int i=0; i<lblock*ntap; i++)  coeff_PPT[i] = 0.;	// initialize array
  for (int k=0; k < ntap; k++){
    for (int j=0; j < lblock; j++){
      for (int i=0; i < ntap - k; i++){
	coeff_PPT[k*lblock + j] += coeff_P[(i+k)*lblock + j] * coeff_P[i*lblock + j];
      }
    }
  } 

  // compute Cholesky factorization of each coeff_PPT submatrix 
  // remember that lapack has row major format
  for (int i=0; i<lblock; i++){
    // band_PPT
    for (int j=0; j<ntap; j++){
      for (int k=0; k<nblock; k++){
        ab[k*ntap + j] = coeff_PPT[(ntap-1-j)*lblock + i];
      }
    }
    dpbtrf_(&uplo, &n, &kd, ab, &ldab, &info); 
    if (info != 0){
      printf("pbtrf error :%d\n",info);
    }
    for (int j=0; j<ntap; j++){
      for (int k=0; k<nblock; k++){
        uPPT[i*ntap*nblock + k*ntap + j] = (float) ab[k*ntap + j];	// cuBLAS also has row major format
      }
    }
  }

  // fill host arrays
  for (int k=0; k<lblock; k++){
    for (int j=0; j<ntap; j++){
      for (int i=0; i<ntsblock; i++){
        band_P[k*ntap*ntsblock + i*ntap + j] = (float) coeff_P[(ntap-1-j)*lblock + k];
      }
    }
  }

  // load to device
  CUDA_CALL( cudaMemcpy(d_uPPT, uPPT, lblock*ntap*nblock*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_band_P, band_P, ntsblock*lblock*ntap*sizeof(float), cudaMemcpyHostToDevice) );

  free(uPPT);
  free(ab);
  free(coeff_P);
  free(coeff_PPT);
  free(band_P);
  return 1;
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
  cufftHandle plan;
  cublasHandle_t handle;
  cublasStatus_t err;
  float *d_pts, *d_uPPT, *d_band_P;
  cudaEvent_t tic,toc;
  float elapsedTime;

  // pull out the number of blocks and their length
  int lblock = 2 * (num_freq - 1);
  int nblock = num_samples / lblock - (num_tap - 1);
  int ntsblock = nblock + num_tap - 1;

  // create CUDA events for timing
  cudaEventCreate(&tic);
  cudaEventCreate(&toc);

  // create cuBLAS context
  CUBLAS_CALL( cublasCreate(&handle) );

  // generate and load coeff_P and Cholesky factorized PPT matrix to device
  CUDA_CALL( cudaMalloc((void **) &d_uPPT, nblock*lblock*num_tap*sizeof(float)) );
  CUDA_CALL( cudaMalloc((void **) &d_band_P, ntsblock*lblock*num_tap*sizeof(float)) );
  PPT(nblock, lblock, num_tap, d_uPPT, d_band_P);

  cudaEventRecord(tic);

  // generate pseudo timestream
  CUDA_CALL( cudaMalloc((void **) &d_pts, nblock*lblock*sizeof(float)) );
  //CUFFT_CALL( cufftPlanMany(&plan, 1, &lblock, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, nblock) );
  int inembed[1] = {(lblock/2+1)};
  int onembed[1] = {lblock};
  //CUFFT_CALL( cufftPlanMany(&plan, 1, &lblock, inembed, 1, lblock/2+1, onembed, 1, lblock, CUFFT_C2R, nblock) );
  CUFFT_CALL( cufftPlanMany(&plan, 1, &lblock, inembed, 1, lblock/2+1, onembed, nblock, 1, CUFFT_C2R, nblock) );
  CUFFT_CALL( cufftExecC2R(plan, d_s, d_pts) );
  CUFFT_CALL( cufftDestroy(plan) );

  // multiple pseudo-timestream by 1./lblock (to rescale inverse FFT)
  dim3 blocks(64,1,1);
  dim3 threads(512,1,1);
  multiply<<<blocks,threads>>>(d_pts,1./lblock,lblock*nblock);

  // probably want to batch this or use streams
  float beta = 0.0,alpha = 1.0;
  for (int i = 0; i < lblock; i++){

    // solve for intermediate vector
    // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-tbsv
    err = cublasStbsv(handle,CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,
		nblock,num_tap-1,
		d_uPPT+i*nblock*num_tap,num_tap,
		d_pts+i*nblock,1);
    if (err != CUBLAS_STATUS_SUCCESS){
      printf("Error at %s:%s:%d\n",__FILE__,__FUNCTION__,__LINE__);
    }

    err = cublasStbsv(handle,CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,
		nblock,num_tap-1,
		d_uPPT+i*nblock*num_tap,num_tap,
		d_pts+i*nblock,1);
    if (err != CUBLAS_STATUS_SUCCESS){
      printf("Error at %s:%s:%d\n",__FILE__,__FUNCTION__,__LINE__);
    }

    // project back onto time-stream
    // http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gbmv
    err = cublasSgbmv(handle,CUBLAS_OP_T,
		nblock, ntsblock, 0, num_tap-1,
		&alpha,d_band_P+i*num_tap*ntsblock,num_tap,
		d_pts+i*nblock, 1, 
		&beta, d_rts+i*ntsblock, 1
	);
    if (err != CUBLAS_STATUS_SUCCESS){
      printf("Error at %s:%s:%d\n",__FILE__,__FUNCTION__,__LINE__);
    }

  }

  cudaEventRecord(toc);
  cudaEventSynchronize(toc);
  cudaEventElapsedTime(&elapsedTime,tic,toc);
  printf("inverse-pfb (gpu only): %f\n",elapsedTime);

  CUDA_CALL( cudaEventDestroy(tic) );
  CUDA_CALL( cudaEventDestroy(toc) );
  CUDA_CALL( cudaFree(d_pts) );
  CUDA_CALL( cudaFree(d_uPPT) );
  CUDA_CALL( cudaFree(d_band_P) );
  CUBLAS_CALL( cublasDestroy(handle) );
  return 1;
}

int main(int argc, char* argv[]){
  int num_beng_frames = 2;
  int num_tap = 4, num_freq = BENG_CHANNELS_ + 1;
  float elapsedTime;
  float *d_ts, *d_rts, *ts, *rts;
  cufftComplex *d_s;
  cudaEvent_t tic, toc;
  curandGenerator_t gen;

  int num_samples = 2*BENG_CHANNELS_*(BENG_SNAPSHOTS*num_beng_frames + num_tap - 1);
  int lblock = 2 * (num_freq - 1);
  int nblock = num_samples / lblock - (num_tap - 1);


  printf("num_samples=%d\n",num_samples);
  printf("num_freqs=%d\n",num_freq);
  printf("lblock=%d\n",lblock);
  printf("nblock=%d\n",nblock);

  // create events
  CUDA_CALL( cudaEventCreate(&tic) );
  CUDA_CALL( cudaEventCreate(&toc) );

  // allocate memory
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

  // it would be good to compute the correlation coefficient here. 

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

  // clean up
  CURAND_CALL( curandDestroyGenerator(gen) );
  CUDA_CALL( cudaEventDestroy(tic) );
  CUDA_CALL( cudaEventDestroy(toc) );
  CUDA_CALL( cudaFree(d_ts) );
  CUDA_CALL( cudaFree(d_s) );
  CUDA_CALL( cudaFree(d_rts) );
  free(ts);
  free(rts);
  fprintf(stdout,"done!\n");
}
