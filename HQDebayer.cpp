#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include "debayer_kernels.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Functions
void cleanup(void);
void initializeData(char *file) ;

// Global variables
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height

int main(int argc, char **argv)
{
  // Helper
  if (checkCmdLineFlag(argc, (const char **)argv, "help"))
  {
      cout << "Usage: debayer <filename>" << endl;
      exit(EXIT_SUCCESS);
  }

  outPixel* data = NULL;
  unsigned char* result = NULL;
  cudaEvent_t start, stop;
  float total_time = 0.0f;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int i = 0;
  while ( i < 10 ) {
    // Initialize data, copy it to device and set up texture memory
    initializeData(argv[1]);
    if (data == NULL)
      checkCudaErrors(cudaMalloc((void**)&data, imWidth * imHeight * sizeof(outPixel)));

    if (result == NULL)
      result = (unsigned char*)malloc(imWidth*imHeight*sizeof(outPixel));
    
    cudaEventRecord(start, 0);
    // Execute Kernel
    hqDebayer(data, imWidth, imHeight);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "Kernel execution time " << time << endl;
    total_time += time;
    cudaEventRecord(start, 0);
    // Copy data from device to host 
    checkCudaErrors(cudaMemcpy(result, data, imWidth*imHeight*sizeof(outPixel), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "Device to Host time " << time << endl;
    cout << endl;
    i++;
  }

  cout << "Average Kernel execution time is " << total_time / 10 << endl;
  Mat image_out(imHeight, imWidth, CV_8UC3, result);
  imwrite("out.jpg", image_out);
  imshow("Debayered", image_out);
  waitKey(0);

  // Free up memeory usage
  checkCudaErrors(cudaFree(data));
  free(result);
  cleanup();
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void cleanup(void)
{
    deleteTexture();
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
