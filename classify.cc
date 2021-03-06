#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"
#include "ta_utilities.hpp"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
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

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    gpuErrChk(cudaEventCreate(&start));         \
    gpuErrChk(cudaEventCreate(&stop));          \
    gpuErrChk(cudaEventRecord(start));          \
}

#define STOP_RECORD_TIMER(name) {                           \
    gpuErrChk(cudaEventRecord(stop));                       \
    gpuErrChk(cudaEventSynchronize(stop));                  \
    gpuErrChk(cudaEventElapsedTime(&name, start, stop));    \
    gpuErrChk(cudaEventDestroy(start));                     \
    gpuErrChk(cudaEventDestroy(stop));                      \
}

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
    // seed generator to 2015
    std::default_random_engine generator(2015);
    std::normal_distribution<float> distribution(0.0, 0.1);
    for (int i=0; i < size; i++) {
        output[i] = distribution(generator);
    }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
    stringstream stream(review_str);
    int component_idx = 0;

    for (string component; getline(stream, component, ','); component_idx++) {
        output[stride * component_idx] = atof(component.c_str());
    }
    assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {
    // TODO: randomly initialize weights. allocate and initialize buffers on
    //       host & device. allocate and initialize streams
    float *weight = (float *) malloc(50 * sizeof(float));
    gaussianFill(weight, 50);
    for(int i = 0; i < 50; i++){
      cout<<"Original weight: "<<weight[i]<<endl;
    }

    float *dev_weight;
    gpuErrChk(cudaMalloc(&dev_weight, 50 * sizeof(float)));

    // Adjust offset to make it like "two" buffers
    float *buffer = (float *) malloc(batch_size * 51 * 2 * sizeof(float));
    float *dev_buffer;
    gpuErrChk(cudaMalloc(&dev_buffer, batch_size * 51 * sizeof(float)));

    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    // main loop to process input lines (each line corresponds to a review)
    int review_idx = 0;
    bool offset = false;
    int offset_size = 0;
    int stream_n = 0;
    for (string review_str; getline(in_stream, review_str); review_idx++) {
        // TODO: process review_str with readLSAReview
        //Check the offset, and adjust values based on this
        if(offset){
          offset_size = batch_size;
          stream_n = 1;
        }
        else{
          offset_size = 0;
          stream_n = 0;
        }
        //Read in a review and store it appropriately in the buffer
        readLSAReview(review_str, &buffer[review_idx % batch_size + offset_size], batch_size);
        // TODO: if you have filled up a batch, copy H->D, call kernel and copy
        //      D->H all in a stream
        if(review_idx != 0 && review_idx % batch_size == 0){
          //Before we start using stream[n] again, ensure that it finishes its previous duties using
          //cudaStreamSynchronize
          if(offset){
            cudaStreamSynchronize(stream[1]);
          }
          else{
            cudaStreamSynchronize(stream[0]);
          }
          //Copy the buffer over the GPU (based on whether or not the offset is currently being used) using stream[n], also based on the offset
          cudaMemcpyAsync(dev_buffer, (void*)(buffer + offset_size * 51), batch_size * 51 * sizeof(float), cudaMemcpyHostToDevice, stream[stream_n]);
          //Print out the errors
          cout << "Errors: " << cudaClassify(dev_buffer, batch_size, .1, weight, stream[stream_n]) << endl;

          //Check that the kernel has run correctly
          cudaError err = cudaGetLastError();
          if  (cudaSuccess != err){
                  cerr << "Error for kernel" << cudaGetErrorString(err) << endl;
          } else {
                  cerr << "No kernel error detected" << endl;
          }
          //flip offset so that the other half of the buffer will be used, ensuring that the memcpy will read the previous values
          offset = !offset;
        }
    }

    // cudaMemcpy(weight, dev_weight, 50 * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: print out weights
    for(int i = 0; i < 50; i++){
      cout<<weight[i]<<endl;
    }
    // TODO: free all memory
    free(weight);
    free(buffer);

    cudaFree(dev_weight);
    cudaFree(dev_buffer);

    cudaStreamSynchronize(stream[0]);
    cudaStreamDestroy(stream[0]);

    cudaStreamSynchronize(stream[1]);
    cudaStreamDestroy(stream[1]);
}

int main(int argc, char** argv) {
    if (argc != 2) {
		printf("./classify <path to datafile>\n");
		return -1;
    }
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 100;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Init timing
	float time_initial, time_final;

    int batch_size = 2048;

	// begin timer
	time_initial = clock();

    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);

	// End timer
	time_final = clock();
	printf("Total time to run classify: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);


}
