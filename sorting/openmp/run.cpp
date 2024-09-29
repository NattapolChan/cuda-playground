#include "defines.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "array_utils.h"
#include "sort.h"
#include <omp.h>

int main(int argc, char *argv[]) {

	double start, seqTime;
	std::vector<int> sequence(N, 0);

	start = omp_get_wtime();
	Sort<int>::bitonic(sequence, N, argv[1]);
	seqTime = omp_get_wtime() - start;

	checkAscending(sequence, N);
	std::cout << "time : " << seqTime << std::endl;

	return 0;
}
