kernel_template = """
/*
 * GPU kernels for the following pipeline components:
 *   VDIF interpreter
 *   B-engine depacketizer
 *   Pre-preprocessor
 *   Re-ordering
 *   Linear interpolation
 */

#include <cufft.h>

__device__ int32_t get_cid_from_vdif(const int32_t *vdif_start)
{
 return (*(vdif_start + 5) & 0x000000FF);
}

__device__ int32_t get_fid_from_vdif(const int32_t *vdif_start)
{
 return (*(vdif_start + 5) & 0x00FF0000)>>16;
}

__device__ int32_t get_bcount_from_vdif(const int32_t *vdif_start)
{
 return ((*(vdif_start + 5)&0xFF000000)>>24) + ((*(vdif_start + 4)&0x00FFFFFF)<<8);
}

__device__ cufftComplex read_complex_sample(int32_t *samples_int)
{
 float sample_imag, sample_real;
  sample_imag = __int2float_rd(*samples_int & 0x03) - 2.0f;
 *samples_int = (*samples_int) >> 2;
  sample_real = __int2float_rd(*samples_int & 0x03) - 2.0f;
 *samples_int = (*samples_int) >> 2;
 return make_cuFloatComplex(sample_real, sample_imag);
}

__global__ void vdif_to_beng(
 int32_t *vdif_frames,
 int32_t *fid_out,
 int32_t *cid_out,
 int32_t *bcount_out,
 cufftComplex *beng_data_out_0,
 cufftComplex *beng_data_out_1,
 int32_t *beng_frame_completion,
 int32_t num_vdif_frames,
 int32_t bcount_offset)
{

 int32_t cid,fid;
 int32_t bcount;
 const int32_t *vdif_frame_start;
 int32_t samples_per_snapshot_half_0, samples_per_snapshot_half_1;
 int32_t idx_beng_data_out;
 int32_t iframe;
 int idata;
 int isample;
 int old;

 for (iframe=0; iframe + threadIdx.y + blockIdx.x*blockDim.y<num_vdif_frames; iframe+=gridDim.x*gridDim.y*blockDim.y)
 {
# 1000 "reader.cu"
  vdif_frame_start = vdif_frames + (iframe + threadIdx.y + blockIdx.x*blockDim.y)*(1056/4);
# 1015 "reader.cu"
   cid = get_cid_from_vdif(vdif_frame_start);
   fid = get_fid_from_vdif(vdif_frame_start);
   bcount = get_bcount_from_vdif(vdif_frame_start);

  cid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = cid;
  fid_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = fid;
  bcount_out[iframe + threadIdx.y + blockIdx.x*blockDim.y] = bcount;
# 1040 "reader.cu"
   idx_beng_data_out = 8 * (cid * 8 + fid)*%(BENG_BUFFER_IN_COUNTS)d*128;
   //idx_beng_data_out += ((bcount-bcount_offset)&(%(BENG_BUFFER_IN_COUNTS)d-1))*128;
   idx_beng_data_out += ((bcount-bcount_offset) %% %(BENG_BUFFER_IN_COUNTS)d)*128; // if BENG_BUFFER_IN_COUNTS is not radix-2
# 1058 "reader.cu"
   idx_beng_data_out += threadIdx.x;
# 1112 "reader.cu"
  for (idata=0; idata<(1024/4); idata+=(8/4)*blockDim.x)
  {
# 1124 "reader.cu"
    samples_per_snapshot_half_0 = *(vdif_frame_start + (32/4) + idata + (8/4)*threadIdx.x);
    samples_per_snapshot_half_1 = *(vdif_frame_start + (32/4) + idata + (8/4)*threadIdx.x + 1);

   for (isample=0; isample<8/2; isample++)
   {

     beng_data_out_1[idx_beng_data_out+(8/2-(isample+1))*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_0);
     beng_data_out_0[idx_beng_data_out+(8/2-(isample+1))*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_0);
     beng_data_out_1[idx_beng_data_out+(8/2-(isample+1)+8/2)*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_1);
     beng_data_out_0[idx_beng_data_out+(8/2-(isample+1)+8/2)*%(BENG_BUFFER_IN_COUNTS)d*128] = read_complex_sample(&samples_per_snapshot_half_1);
# 1193 "reader.cu"
   }
    idx_beng_data_out += blockDim.x;
  }
# 1253 "reader.cu"
  old = atomicAdd(beng_frame_completion + ((bcount-bcount_offset) %% %(BENG_BUFFER_IN_COUNTS)d), 1);
# 1277 "reader.cu"
  if (__any(old == ((16384/8)*blockDim.x)-1))
  {
# 1298 "reader.cu"
  }
 }
}


__global__ void reorderTz_smem(cufftComplex *beng_data_in, cufftComplex *beng_data_out, int num_beng_frames){
  // gridDim.x = 16384 * 128 / (16 * 16) = 8192
  // blockDim.x = 16; blockDim.y = 16;
  // --> launches 2097152 threads

  int32_t sid_out,bid_in;

  __shared__ cufftComplex tile[16][16];

  // for now, let us loop the grid over B-engine frames:
  for (int bid_out=0; bid_out<num_beng_frames-1; bid_out+=1){

    // input snapshot id
    int sid_in = (blockIdx.x * blockDim.x + threadIdx.x) %% 128;
    // input channel id 
    int cid = threadIdx.y + blockDim.y * (blockIdx.x / (128 / blockDim.x));

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    if (sid_out < 69){
      bid_in = bid_out;
    } else {
      bid_in = bid_out+1;
    }

    tile[threadIdx.x][threadIdx.y] = beng_data_in[128*num_beng_frames*cid + 128*bid_in + sid_in];

    __syncthreads();

    // now we tranpose warp orientation over channels and snapshot index

    // snapshot id
    sid_in = threadIdx.y + (blockIdx.x*blockDim.y) %% 128;
    // channel id 
    cid = threadIdx.x + blockDim.x * (blockIdx.x / (128 / blockDim.x)); 

    // shift by 2-snapshots case:
    if (((cid / 4) & (0x1)) == 0) {
      sid_out = (sid_in-2) & 0x7f;
    } else {
      sid_out = sid_in;
    }

    beng_data_out[128*16385*bid_out + 16385*sid_out + cid] = tile[threadIdx.y][threadIdx.x];

    // zero out nyquist: 
    if (cid == 0) {
      beng_data_out[128*16385*bid_out + 16385*sid_out + 16384] = make_cuComplex(0.,0.);
    }

    __syncthreads();
  }
}

__global__ void nearest(float *a, int Na, float *b, int Nb, double c){
  /*
  This kernel uses a round-half-to-even tie-breaking rule which is
  opposite that of python's interp_1d.
  a: input_array
  b: output_array
  Nb: size of array b
  c: stride for interpolation: b[i] = a[int(c*i)]
  */
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < Nb) {
    int ida = __double2int_rn(tid*c); // round nearest
      b[tid] = a[ida];
  }
}

__global__ void linear1(const float *a, float *b, const int N, const float *wgt, const int *ida){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = ida[tid];
  float w = wgt[tid];
  // loop over groups of 39 snapshots (assuming we have 128 such chunks)
  for (int ichunk=0; ichunk<128; ichunk += 1){
    if (i+1+1277952*ichunk<N){
      b[ichunk*2097152+tid] = a[i+1277952*ichunk]*(1.f-w) + a[i+1+1277952*ichunk]*w;
    } else {
      b[ichunk*2097152+tid] = a[i+1277952*ichunk];
    }
  }
}

__global__ void linear(float *a, int Na, float *b, int Nb, double c){
 /*
  a: input_array (assume padded by two floats for every SWARM snapshot)
  b: output_array
  Nb: size of array b
  c: conversion factor between a and b indices. 
  Note: type conversions are slowing this down.
  Idea: Use texture memory to store ida and weights.
  */

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < Nb) {
    int ida = __double2int_rd(tid*c); // round down
    if (ida < Na-1){
      //b[tid] = d * ( a[(ida / 32768)*32770 + (ida %% 32768) ]*(1.-(c*tid-ida)) + 
//		     a[((ida+1) / 32768)*32770 + ((ida+1) %% 32768)]*(c*tid-ida) );
      b[tid] = ( a[ida]*(1.-(c*tid-ida)) + a[ida+1]*(c*tid-ida) );
    } else {
      //b[tid] = d * a[(ida / 32768)*32770 + (ida %% 32768) ];
      b[tid] = a[ida];
    }
  }
}

__global__ void zero_cout(cufftComplex *a, int32_t n)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n){
    a[tid] = make_cuComplex(0.,0.);
  }
}

__global__ void zero_rout(float *a, int32_t n)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n){
    a[tid] = 0.;
  }
}

__global__ void strided_copy(float *a, int istart, float *b, int N, int istride, int iskip)
{
  int32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < N){
    b[tid] = a[istart+(tid/istride)*(istride+iskip) + (tid %% istride) ];
  }
}

__global__ void quantize2bit(const float *in, unsigned int *out, int N, float thresh)
{
/*
 * 2bit quantization kernel. Has to be called with 16 x-threads and any 
 * number of y-threads, and any number of x-blocks
 */
	int idx_in = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int idx_out = blockIdx.x*blockDim.y + threadIdx.y;
	
	for (int ii=0; (idx_in+ii)<N; ii+=gridDim.x*blockDim.x*blockDim.y)
	{
		//This is is for 00 = -2, 01 = -1, 10 = 0, 11 = 1.
		/* Assume sample x > 0, lower bit indicates above threshold. Then
		 * test, if x < 0, XOR with 11.
		 */
		int sample_2bit = ( ((fabsf(in[idx_in+ii]) >= thresh) | 0x02) ^ (0x03*(in[idx_in+ii] < 0)) ) & 0x3;
		//~ //This is for 11 = -2, 10 = -1, 01 = 0, 10 = 1
		//~ int sample_2bit = ((fabsf(in[idx_in+ii]) <= thresh) | ((in[idx_in+ii] < 0)<<1)) & OUPUT_MAX_VALUE_MASK;
		sample_2bit = sample_2bit << (threadIdx.x*2);
		atomicOr(out+idx_out, sample_2bit);
		idx_out += gridDim.x*blockDim.y;
	}
}
"""
