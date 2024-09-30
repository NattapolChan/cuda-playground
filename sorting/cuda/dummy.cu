#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// blockDim.x - num threads in a block, .x indicates 1D block labelling
// blockIdx.x - thread index number
// multiplying the above two variables gives start of block // then add the threadIdx.x offset for the particular thread

__global__ void saxpy_parallel(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n)  y[i] = a*x[i] + y[i];
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	std::cout << "device count = " << deviceCount << std::endl;

	int N = 256;
	// allocate vectors on host
	int size = N * sizeof(float);
	float* h_x = (float*)malloc(size);
	float* h_y = (float*)malloc(size);
	
	for (int i = 0;i<=N-1;i++)
	{
		h_x[i]=4;
		h_y[i]=2;
	}

	for (int i = 0;i<=N-1;i++)
	{
		std::cout << i << " " <<  h_y[i] << std::endl;
	}
	

	// allocate device memory
	float* d_x; float* d_y;

	checkCudaError(cudaMalloc((void**) &d_x, size), "malloc1");
	checkCudaError(cudaMalloc((void**) &d_y, size), "malloc2");

	checkCudaError(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice), "memcpy 1");
	checkCudaError(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice), "memcpy 2");

	// calculate number of blocks needed for N 
	int nblocks = (N+255)/256;

	// call 
	saxpy_parallel<<<nblocks,256>>>(N,2.0,d_x,d_y);

	checkCudaError(cudaGetLastError(), "kernel launch");
	
	// Copy results back from device memory to host memory
	// implicty waits for threads to excute
	checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	checkCudaError(cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost), "memcpy 3");

	for (int i = 0;i<=N-1;i++)
	{
		std::cout << i << " last loop " <<  h_y[i] << std::endl;
	}



	cudaFree(d_x);
	cudaFree(d_y);

	free(h_x);
	free(h_y);



	return 0;

}
