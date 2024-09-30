#include "defines.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "array_utils.h"
#include <stdio.h>
#include "cuda_runtime.h"
#include <chrono>

__global__ void parallelBitonicSwap(int* a, uint64_t size, uint64_t i, uint64_t j) { 
	uint64_t k = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= size) return;
	if ((k ^ (1 << j)) > k) {
		if (((1 << i) & k) == 0) {
			if (a[k] > a[k ^ (1 << j)]) {
				int tmp = a[k];
				a[k] = a[k ^ (1 << j)];
				a[k ^ (1 << j)] = tmp;
			}
		} else {
			if (a[k] < a[k ^ (1 << j)]) {
				int tmp = a[k];
				a[k] = a[k ^ (1 << j)];
				a[k ^ (1 << j)] = tmp;
			}
		}
	}
}

void bitonicCuda(std::vector<int>& a, uint64_t size) {
	int* arr = &a[0];
	int* d_arr;

	cudaMalloc((void**) &d_arr, size*sizeof(int));
	cudaMemcpy(d_arr, arr, size*sizeof(int), cudaMemcpyHostToDevice);
	for (uint64_t i=1;i<=std::log2(size);i++) {
		for(long long j=i-1;j>=0;j--) {
			parallelBitonicSwap<<<(uint64_t)size/65535, 65535>>>(d_arr, size, i, j);
		}
	}

	cudaMemcpy(arr, d_arr, size*sizeof(int), cudaMemcpyDeviceToHost);

	for (uint64_t i=0;i<size;i++) a[i] = arr[i];

	// free(arr);
	// cudaFree(d_arr);
}

int main(int argc, char *argv[]) {

	std::vector<int> sequence(N, 0);
	Generator<int>::random(sequence, N);

	float gpuTime = 0;

	auto startTime = std::chrono::high_resolution_clock::now();

	bitonicCuda(sequence, N);

    	auto endTime = std::chrono::high_resolution_clock::now();
    	gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    	gpuTime = gpuTime / 1000000;
	
	printf("Sorting %lld elements in %.4f seconds\n", (long long) N, gpuTime);

	checkAscending(sequence, N);

	return 0;
}
