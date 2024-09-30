#include <iostream>
#include "defines.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cassert>

template<typename T>
static void checkAscending(const std::vector<T> &a, uint64_t size) {
	for (uint64_t i=0;i<size-1;i++)
		if (a[i] > a[i+1]) {
			std::cout << a[i] << " should be less or equal " << a[i+1] << " | Index i = " << i << std::endl;
			assert(false);
		}
};

template<typename T>
static void checkDescending(const std::vector<T> &a, uint64_t size) {
	for (uint64_t i=0;i<size-1;i++)
		assert(a[i] <= a[i+1]);
};

template<typename T> 
class Generator {
	public:
		static void random(std::vector<T> &a, uint64_t size);
		static void random(std::vector<T> &a, uint64_t size, long long maxNumber);
		static void decreasing(std::vector<T> &a, uint64_t size);
		static void increasing(std::vector<T> &a, uint64_t size);
};

template<typename T>
void Generator<T>::random(std::vector<T> &a, uint64_t size) {
	srand (time(NULL));
	for (uint64_t i=0;i<size;i++)
		a[i] = (T)rand() % (MAX - MIN) + MIN;
}

template<typename T>
void Generator<T>::increasing(std::vector<T> &a, uint64_t size) {
	srand (time(NULL));
	for (uint64_t i=0;i<size;i++)
		a[i] = rand() % (MAX - MIN) + MIN;
}
