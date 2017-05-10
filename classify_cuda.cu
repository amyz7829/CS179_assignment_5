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
      float* gradient = &shmem[sizeof(float) * 50];

      float *x = (float *)malloc(51 * sizeof(float));
      for(int i = 0; i < 51; i++){
        x[i] = data[idx + i * batch_size];
      }
      float *grad = (float *)malloc(50 * sizeof(float));
      //The error value is the dot product
      float error_val = 0;
      for(int i = 0; i < 50; i++){
        error_val += weight_v[i] * x[i];
      }
      // If there is an error, add to the error
      if(error_val <= 0){
        *errors += 1 / batch_size;
      }
      //Find the gradient for the data point x
      for(int i = 0; i < 50; i++){
        grad[i] = (1 / batch_size) * (x[50] * x[i]) / (1 + exp(1 + x[50] * error_val));
      }

      //Now atomically build gradient
      for(int i = 0; i < 50; i++){
        atomicAdd(&gradient[i], grad[i]);
      }

      //Only one thread needs to subtract
      if(idx == 0){
        for(int i = 0; i < 50; i++){
          weights[i] = weight_v[i] - step_size * grad[i];
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
    int block_size = (batch_size < 1024) ? batch_size : 1024;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = sizeof(float) * 100;

    float *d_errors;
    cudaMalloc(&d_errors, sizeof(float));
    cudaMemset(d_errors, 0, sizeof(float));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);

    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}
