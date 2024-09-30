#include "array_utils.h"
#include "defines.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
