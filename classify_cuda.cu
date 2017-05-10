#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
 #define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
 inline void gpuAssert(
     cudaError_t code,
     const char *file,
     int line,
     bool abort=true)
 {
     if (code != cudaSuccess) {
         fprintf(stderr,"GPUassert: %s %s %d\n",
             cudaGetErrorString(code), file, line);
         exit(code);
     }
 }

__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    int step_size,
	float *weights,
    float *errors)
{
    // TODO: write me
    int tid = threadIdx.x;
    uint idx = blockIdx.x * blockDim.x + tid;
    while(idx < batch_size){
      extern __shared__ float shmem[];
      float* weight_v = &shmem[0];
      weight_v = weights;

      float* gradient = &shmem[50];

      //The error value is the dot product
      float error_val = 0;
      for(int i = 0; i < 50; i++){
        error_val += weight_v[i] * data[idx + i * batch_size];
      }
      // If there is an error, add to the error
      if(error_val <= 0){
        atomicAdd(errors, 1 / batch_size);
      }
      //Find the gradient for the data point x
      for(int i = 0; i < 50; i++){
        assert(idx + 51 * batch_size < batch_size * 51);
        atomicAdd(&gradient[i], (1 / batch_size) * (data[idx + 51 * batch_size] * data[idx + i * batch_size]) / (1 + exp(1 + data[idx + i * batch_size] * error_val)));
      }

      //Only one thread needs to subtract
      if(idx == 0){
        for(int i = 0; i < 50; i++){
          weights[i] = weight_v[i] - step_size * gradient[i];
        }
      }
      idx += gridDim.x * blockDim.x;
    }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float *data,
    int batch_size,
    float step_size,
    float *weights,
    cudaStream_t stream)
{
    // int block_size = (batch_size < 1024) ? batch_size : 1024;
    int block_size = 512;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;

    // 50 floats for the weight, 50 floats for the gradient
    int shmem_bytes = sizeof(float) * 100;

    float *d_errors;
    gpuErrChk(cudaMalloc(&d_errors, sizeof(float)));
    gpuErrChk(cudaMemset(d_errors, 0, sizeof(float)));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);

    float h_errors = -1.0;
    gpuErrChk(cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault));
    gpuErrChk(cudaFree(d_errors));
    return h_errors;
}
