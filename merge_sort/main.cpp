#include<iostream>
#include<vector>
#include<random>
#include<omp.h>
#include<cassert>

using namespace std;

vector<int> merge_sort_extra_space(vector<int> a, int start, int end);
void quick_sort(vector<int>& a, int pivotIdx, int pivotEnd);
void mergeSortOpenmp(vector<int>& a, int start, int end);

template <typename T>
vector<T> generateRandomVector(int size, T minValue, T maxValue) {
	vector<T> vec(size);
	srand(static_cast<unsigned>(time(0)));
	for (int& num : vec)
		num = minValue + rand() % (maxValue - minValue + 1);
	return vec;
}

template <typename T>
bool isSortedIncreasing(const vector<T>& vec) {
    for (size_t i = 1; i < vec.size(); ++i)
        if (vec[i - 1] > vec[i])
            return false;
    return true;
}

int main() {
	vector<int> first = generateRandomVector<int>(1000, 0, 100.0);
	vector<int> second = generateRandomVector<int>(5000, 0, 100.0);

	vector<int> answerf, answers;

	double start = omp_get_wtime();
	answerf = merge_sort_extra_space(first, 0, first.size());
	answers = merge_sort_extra_space(second, 0, second.size());
	double seqTime = omp_get_wtime() - start;
	cout << seqTime << endl;

	start = omp_get_wtime();
	mergeSortOpenmp(first, 0 , first.size());
	mergeSortOpenmp(second, 0, second.size());
	seqTime = omp_get_wtime() - start;
	cout << seqTime << endl;

	assert(isSortedIncreasing(first));
	assert(isSortedIncreasing(second));
	assert(isSortedIncreasing(answerf));
	assert(isSortedIncreasing(answers));
}

// O(n) extra space merge sort 
vector<int> merge_sort_extra_space(vector<int> a, int start, int end) {
	int mid =  ( ( start + end ) >> 1 ) + 1;
	if (end - start < 3) {
		vector<int> b{
			min(a[start], a[end-1]), 
			max(a[start], a[end-1])
		};
		b.push_back(min(a[start], a[end-1]));
		b.push_back(max(a[start], a[end-1]));
		return b;
	}
	vector<int> fh = merge_sort_extra_space(a, start, mid);
	vector<int> sh = merge_sort_extra_space(a, mid, end);

	vector<int> newa;

	int fp = 0, sp = 0;
	
	while (fp < mid - start && sp < end - mid) {
		if (fh[fp] < sh[sp]) newa.push_back(fh[fp++]);
		else newa.push_back(sh[sp++]);
	}
	while (fp < mid - start) newa.push_back(fh[fp++]);
	while (sp < end - mid) newa.push_back(sh[sp++]);

	return newa;
}

// 1 3 0 4 2 5 8
// pivot = 0
// 0 1 3 4 2 5 8
// pivot = prevLessThanPivot
// 0 1 | 3 4 2 5 8
// 0 1 | 2 3 4 5 8
// pivot = prevLessThanPivot

void quick_sort(vector<int>& a, int pivotIdx, int pivotEnd) {
	if (pivotIdx >= pivotEnd) return;
	int pivot = a[pivotIdx];
	int fp=pivotIdx + 1;
	int sp=pivotEnd;

	while (fp < sp) {
		if (a[fp] < pivot) fp++;
		else { 
			int tmp=a[fp];
			a[fp]=a[sp];
			a[sp]=tmp; sp--;
		}
		// for( int x : a ) cout << x << " ";
		// cout << '\n';
	}
	cout << pivotIdx << " " << pivot << '\n';
	for( int x : a ) cout << x << " ";
	cout << '\n';
	int tmp=a[pivotIdx];
	a[pivotIdx]=a[fp-1];
	a[fp-1]=tmp;
	quick_sort(a, fp, pivotEnd);
	quick_sort(a, pivotIdx, fp-1);
}

void mergeSortOpenmp(vector<int>& a, int start, int end) {
	if (end - start <= 1) return;
	int mid = ((start + end) >> 1);

#pragma omp parallel 
{
	#pragma omp single 
	{
		#pragma omp task
			mergeSortOpenmp(a, start, mid);

		#pragma omp task
			mergeSortOpenmp(a, mid, end);

		#pragma omp taskwait
	}
}

	vector<int> left(a.begin() + start, a.begin() + mid);
	vector<int> right(a.begin() + mid, a.begin() + end);

	int fp= 0, sp=0, cur=start;

	while (fp < left.size() && sp < right.size()) {
	    if (left[fp] < right[sp])
	        a[cur++] = left[fp++];
	    else 
	        a[cur++] = right[sp++];
	}
	while (fp < left.size())
	    a[cur++] = left[fp++];
	while (sp < right.size())
	    a[cur++] = right[sp++];
}
