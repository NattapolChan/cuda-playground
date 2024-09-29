#include <stdio.h>
#include <vector>
#include "defines.h"
#include <cassert>
#include <cmath>
#include <assert.h>
#include <string>
#include <map>
#include <iostream>
#ifdef __NVCC__
#include "cuda_runtime.h"
#endif

template<typename T>
class Sort {
public:
	static void bitonic(std::vector<T>& a, uint32_t size, char* mode);
	static void merge(std::vector<T>& a, uint32_t size);
};

template<typename T>
void bitonicOpenMP(std::vector<T>& a, uint32_t size) {
	for (uint32_t i=1;i<=std::log2(size);i++) {
		for(int j=i-1;j>=0;j--) {
			#pragma omp parallel for shared(a)
			for (int k=0;k<size;k++) {
				if ((k ^ (1 << j)) > k) {
					if (((1 << i) & k) == 0) {
						if (a[k] > a[k ^ (1 << j)]) std::swap(a[k], a[k ^ (1 << j)]);
					} else {
						if (a[k] < a[k ^ (1 << j)]) std::swap(a[k], a[k ^ (1 << j)]);
					}
				}
			}
		}	
	}
}

template<typename T>
void bitonicSequential(std::vector<T>& a, uint32_t size) {

	for (uint32_t i=1;i<=std::log2(size);i++) {
		for(int j=i-1;j>=0;j--) {
			for (int k=0;k<size;k++) {
				if ((k ^ (1 << j)) > k) {
					if (((1 << i) & k) == 0) {
						if (a[k] > a[k ^ (1 << j)]) std::swap(a[k], a[k ^ (1 << j)]);
					} else {
						if (a[k] < a[k ^ (1 << j)]) std::swap(a[k], a[k ^ (1 << j)]);
					}
				}
			}
		}	
	}
};

#ifdef __NVCC__
template<typename T>
__global__ void parallelBitonicSwap(T* a, uint32_t size, uint32_t i, uint32_t j) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if ((k ^ (1 << j)) > k) {
		if (((1 << i) & k) == 0) {
			if (a[k] > a[k ^ (1 << j)]) std::swap(a[k], a[k ^ (1 << j)]);
		} else {
			if (a[k] < a[k ^ (1 << j)]) std::swap(a[k], a[k ^ (1 << j)]);
		}
	}
}
#endif

template<typename T>
void bitonicCuda(std::vector<T>& a, u_int32_t size) {
	T* arr = &a[0];

	cudaMallocManaged(&arr, size*sizeof(T));
	for (uint32_t i=1;i<=std::log2(size);i++) {
		for(int j=i-1;j>=0;j--) {
			#ifdef __NVCC__
				parallelBitonicSwap<<<size/256, 256>>>(a, size, i, j);
			#endif
		}	
	}
}

constexpr int hash(const char *str) {
	long hashVal = 5381;
	int ch = 0;
	while (ch = *str++) hashVal = ((hashVal << 5) + hashVal) + ch;
	return hashVal;
}

template<typename T>
void Sort<T>::bitonic(std::vector<T>& a, uint32_t size, char* mode) {

	assert(!(size & (size - 1))); // check power of 2
	switch(hash(mode)) {
		case hash("SEQUENTIAL"):
			bitonicSequential(a, size);
			break;
		case hash("OPENMP"):
			bitonicOpenMP(a, size);
			break;
		case hash("CUDA"):
			#ifdef __NVCC__
				bitonicCuda(a, size);
			#else
				std::cout << "Use nvcc to compile instead" << std::endl;
			#endif
			break;
		default:
			std::cout << "Should not enter here" << std::endl;
			assert(false);
			break;
	}
};

