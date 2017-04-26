#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_string.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include "debayer_kernels.h"
// Texture reference for reading image
texture<unsigned char, 2> tex;
static cudaArray *array = NULL;

__global__ void
DebayerTex(outPixel *pOut, unsigned int Pitch, int w, int h, bool offset_x, bool offset_y)
{
  outPixel *pDebayer = (outPixel *)(pOut + blockIdx.x * Pitch);
	const float fs1_a = -0.125f, fs1_b = -0.125f,  fs1_c =  0.0625f, fs1_d = -0.1875f;
	const float fs2_a = -0.125f, fs2_b =  0.0625f, fs2_c = -0.125f , fs2_d = -0.1875f;
	const float fs3_a =  0.25f , fs3_b =  0.5f;
	const float fs4_a =  0.25f , fs4_c =  0.5f;
	const float fs5_b = -0.125f, fs5_c = -0.125f, fs5_d =  0.25f;
	const float fs6_a = 0.5f , fs6_b = 0.625f, fs6_c = 0.625f, fs6_d = 0.75f;

  for (int i = threadIdx.x; i < w; i += blockDim.x)
  {
    float s1 = tex2D(tex, (float) i - 2, (float)blockIdx.x) + tex2D(tex, (float) i + 2, (float) blockIdx.x);
    float s2 = tex2D(tex, (float) i, (float)blockIdx.x - 2) + tex2D(tex, (float) i, (float) blockIdx.x + 2);
    float s3 = tex2D(tex, (float) i - 1, (float)blockIdx.x) + tex2D(tex, (float) i + 1, (float) blockIdx.x);
    float s4 = tex2D(tex, (float) i, (float)blockIdx.x - 1) + tex2D(tex, (float) i, (float) blockIdx.x + 1);
    float s5 = tex2D(tex, (float) i - 1, (float)blockIdx.x - 1) + tex2D(tex, (float) i + 1, (float) blockIdx.x - 1) + 
               tex2D(tex, (float) i - 1, (float) blockIdx.x + 1) + tex2D(tex, (float) i + 1, (float) blockIdx.x + 1);
    float s6 = tex2D(tex, (float) i, (float)blockIdx.x);
    
    float filter_a = fs1_a*s1 + fs2_a*s2 + fs3_a*s3 + fs4_a*s4 + 0.0f     + fs6_a*s6;
	  float filter_b = fs1_b*s1 + fs2_b*s2 + fs3_b*s3 + 0        + fs5_b*s5 + fs6_b*s6;
	  float filter_c = fs1_c*s1 + fs2_c*s2 + 0        + fs4_c*s4 + fs5_c*s5 + fs6_c*s6;
	  float filter_d = fs1_d*s1 + fs2_d*s2 + 0 + 0 + fs5_d*s5 + fs6_d*s6;
    
	  if (((i + offset_x) % 2 == 0) && ((blockIdx.x + offset_y) % 2 == 0)) { // 0,0
	    pDebayer[i].B = (unsigned char)s6;
	  	pDebayer[i].G = (unsigned char)(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
	  	pDebayer[i].R = (unsigned char)(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
	  }
	  else if (((i + offset_x) % 2 == 1) && ((blockIdx.x + offset_y) % 2 == 0)) { // 1,0
	  	pDebayer[i].G = (unsigned char)s6;
	  	pDebayer[i].B = (unsigned char)(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
	  	pDebayer[i].R = (unsigned char)(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
	  }
	  else if (((i + offset_x) % 2 == 0) && ((blockIdx.x + offset_y) % 2 == 1)) { // 0,1
	  	pDebayer[i].G = (unsigned char)s6;
	  	pDebayer[i].B = (unsigned char)(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
	  	pDebayer[i].R = (unsigned char)(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
	  }
	  else { // 1,1
	  	pDebayer[i].R = (unsigned char)s6;
	  	pDebayer[i].B = (unsigned char)(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
	  	pDebayer[i].G = (unsigned char)(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
	  }
  }
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp)
{

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaChannelFormatDesc desc;

  if (Bpp == 1)
  {
    desc = cudaCreateChannelDesc<unsigned char>();
  }
  else
  {
    desc = cudaCreateChannelDesc<uchar4>();
  }
  
  if (array == NULL) {
    cudaEventRecord(start, 0);
    checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Malloc time %f\n", time);
  } 

  cudaEventRecord(start, 0);
  checkCudaErrors(cudaMemcpyToArray(array, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Host to Device %f ms\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

extern "C" void deleteTexture(void)
{
  checkCudaErrors(cudaFreeArray(array));
}

extern "C" void hqDebayer(outPixel *odata, int iw, int ih)
{
  checkCudaErrors(cudaBindTextureToArray(tex, array));

  DebayerTex<<<ih, 384>>>(odata, iw, iw, ih, false, false);

  checkCudaErrors(cudaUnbindTexture(tex));
}
