#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include "debayer_kernels.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Functions
void cleanup(void);
void initializeData(char *file) ;
void computeTime();

// Global variables
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height
StopWatchInterface *timer = NULL;

int main(int argc, char **argv)
{
  // Helper
  if (checkCmdLineFlag(argc, (const char **)argv, "help"))
  {
      cout << "Usage: debayer <filename>" << endl;
      exit(EXIT_SUCCESS);
  }

  // Initialize data, copy it to device and set up texture memory
  initializeData(argv[1]);
  outPixel* data = NULL;
  checkCudaErrors(cudaMalloc((void**)&data, imWidth * imHeight * sizeof(outPixel)));

  if (data == NULL) {
    cout << "Cannot allocate CUDA memory" << endl;
    return 0; 
  }

  // Timer
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  
  hqDebayer(data, imWidth, imHeight);
  checkCudaErrors(cudaDeviceSynchronize());
  
  sdkStopTimer(&timer);
  computeTime();
 
  // Copy data from device to host 
  unsigned char* result = (unsigned char*)malloc(imWidth*imHeight*sizeof(outPixel));
  checkCudaErrors(cudaMemcpy(result, data, imWidth*imHeight*sizeof(outPixel), cudaMemcpyDeviceToHost));
  
  Mat image_out(imHeight, imWidth, CV_8UC3, result);
  imwrite("out.jpg", image_out);

  // Free up memeory usage
  checkCudaErrors(cudaFree(data));
  free(result);
  cleanup();
}

void computeTime()
{
    cout << sdkGetTimerValue(&timer) << endl;
}

void cleanup(void)
{
    deleteTexture();
    sdkDeleteTimer(&timer);
}

void initializeData(char *file)
{
  unsigned char *pixels = NULL;  // Image pixel data on the host
  unsigned int g_Bpp;
  
  Mat image = imread(file, -1);
  imWidth = image.cols;
  imHeight = image.rows;
  g_Bpp = image.channels();
  pixels = image.data;
  setupTexture(imWidth, imHeight, pixels, g_Bpp);

  memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * imWidth * imHeight);
}