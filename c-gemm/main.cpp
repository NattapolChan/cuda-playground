#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <ctime>

template<typename T> class MatrixBase {
public: 
	virtual ~MatrixBase() = default;
	virtual T& operator() (int i, int j) = 0;
	virtual const T& operator() (int i, int j) const = 0;

	virtual int getRow() const = 0;
	virtual int getCol() const = 0;

	virtual void fillFixedValue(T value) = 0;
	virtual void fillRandomValue(T minValue, T maxValue) = 0;
	void print() const {
		for (int i=0;i<(*this).getRow();i++) {
			std::cout << " > [LOGGING] ";
			for (int j=0;j<(*this).getCol();j++)
				std::cout << (*this) (i, j) << " ";
			std::cout << std::endl;
		}
	}
	virtual void transpose() {};
};

template <typename T>
class Matrix : public MatrixBase<T> {
private:
	std::vector<std::vector<T>> matrix;
	int numRows, numCols;

public:
	Matrix(int N): numRows(N), numCols(N), matrix(N, std::vector<T>(N)) {}
	Matrix(int row, int col): numRows(row), numCols(col), matrix(row, std::vector<T> (col)) {}

	T& operator() (int i, int j) override {
		assert(i < numRows && j < numCols && i>=0 && j>=0);
		return matrix[i][j];
	}

	const T& operator() (int i, int j) const override {
		assert(i < numRows && j < numCols && i>=0 && j>=0);
		return matrix[i][j];
	}

	int getRow() const override {return numRows;}
	int getCol() const override {return numCols;}

	void fillFixedValue(T value) override {
		for (int i=0;i<numRows;i++)
			std::fill(matrix[i].begin(), matrix[i].end(), value);
	}

	void fillRandomValue(T minValue, T maxValue) override {
		for (int i=0;i<numRows;i++)
			for (int j=0;j<numCols;j++)
				matrix[i][j] = rand() % (maxValue - minValue) + minValue;
	}

	// try change to inplace
	void transpose() override {
		std::vector<std::vector<T>> tmpMatrix(matrix);
		matrix.resize(numCols, std::vector<T>(numRows));
		for (int i=0;i<numRows;i++)
			for (int j=0;j<numCols;j++)
				matrix[j][i] = tmpMatrix[i][j];
		// std::swap(numRows, numCols);
		int tmp = numRows;
		numRows = numCols;
		numCols = tmp;
	}
};

template <typename T>
class MatrixFlatten : public MatrixBase<T> {
private:
	std::vector<T> matrix;
	int numRows, numCols;

public:
	MatrixFlatten(int N): numRows(N), numCols(N), matrix(N * N) {}
	MatrixFlatten(int row, int col): numRows(row), numCols(col), matrix(row * col) {}

	T& operator() (int i, int j) override {
		assert(i < numRows && j < numCols && i>=0 && j>=0);
		return matrix[numCols * i + j];
	}

	const T& operator() (int i, int j) const override {
		assert(i < numRows && j < numCols && i>=0 && j>=0);
		return matrix[numCols * i + j];
	}

	int getRow() const override {return numRows;}
	int getCol() const override {return numCols;}

	void fillFixedValue(T value) override {
		std::fill(matrix.begin(), matrix.end(), value);
	}

	void fillRandomValue(T minValue, T maxValue) override {
		for (int i=0;i<numRows;i++)
			for (int j=0;j<numCols;j++)
				(*this) (i, j) = rand() % (maxValue - minValue) + minValue;
	}

	// try change to inplace
	void transpose() override {
		std::vector<T> tmpMatrix(matrix);
		for (int i=0;i<numCols;i++)
			for (int j=0;j<numRows;j++)
				matrix[i * numRows + j] = tmpMatrix[j * numCols+ i];
		std::swap((*this).numRows, (*this).numCols);
	}
};

enum EnumMatrixMatrixMultiplicationMethod {
	SEQUENTIAL=0,
	SEQUENTIALTRANSPOSE=1,
	OPENMP=2,
	PARTITIONEDOPENMP=3,
};

template <typename T>
void gemmSequential(const MatrixBase<T>& A, const MatrixBase<T>& B, MatrixBase<T>& C) {
	for (int i = 0; i < A.getRow(); i++) {
		for (int j = 0; j < B.getCol(); j++) {
			C(i, j) = 0;
			for (int k = 0; k < A.getCol(); k++) {
				T a = A(i, k);
				T b = B(k, j);
				C(i, j) += a * b;
			}
		}
	}
}

template <typename T>
void gemmSequentialTranspose(const MatrixBase<T>& A, MatrixBase<T>& B, MatrixBase<T>& C) {
	B.transpose();
	for (int i = 0; i < A.getRow(); ++i) {
		for (int j = 0; j < B.getRow(); ++j) {
			C(i, j) = 0;
			for (int k = 0; k < A.getCol(); ++k) {
				T a = A(i, k);
				T b = B(j, k);
				C(i, j) += a * b;
			}
		}
	}
	B.transpose();
}

template <typename T>
void gemmOpenmp(const MatrixBase<T>& A, const MatrixBase<T>& B, MatrixBase<T>& C) {
#pragma omp parallel for 
	for (int j = 0; j < B.getCol(); ++j) {
		for (int i = 0; i < A.getRow(); ++i) {
			C(i, j) = 0;
			for (int k = 0; k < A.getCol(); ++k) {
				T a = A(i, k);
				T b = B(k, j);
				C(i, j) += a * b;
			}
		}
	}
}

/*
 * Still does not work for (A.size() % blockSize > 0)
 * TODO: add padding matrix before multiplication
 */
template <typename T>
void gemmPartitionedOpenmp(const MatrixBase<T>& A, const MatrixBase<T>& B, MatrixBase<T>& C, int blockSize=64) {

	assert(A.getRow() % blockSize == 0);
	assert(B.getRow() % blockSize == 0);
	assert(C.getCol() % blockSize == 0);

	int blockMNumber = A.getRow() / blockSize;
	int blockNNumber = B.getRow() / blockSize;
	int blockKNumber = C.getCol() / blockSize;

#pragma omp parallel for 
	for (int bi=0;bi<blockMNumber;bi++) {
	for (int bj=0;bj<blockNNumber;bj++) {
	for (int bk=0;bk<blockKNumber;bk++) {
		for (int i=0;i<blockSize;i++) {
		for (int j=0;j<blockSize;j++) {
			T cumulative = 0;
			for (int k=0;k<blockSize;k++)
				cumulative += A(bi*blockSize + i, bk*blockSize + k) * B(bk*blockSize + k,bj*blockSize + j);
			C(bi*blockSize+i, bj*blockSize+j) += cumulative;
		}
	}}}}
}

template <typename T>
void gemm(MatrixBase<T>& A, MatrixBase<T>& B, MatrixBase<T>& C, const EnumMatrixMatrixMultiplicationMethod method, int logLevel = 3) {
	assert (A.getCol() == B.getRow() && C.getRow() == A.getRow() && C.getCol() == B.getCol());

#pragma omp parallel for 
	for (int i=0;i<C.getRow();i++)
		for (int j=0;j<C.getCol();j++) 
			C(i,j) = 0;

	double startTime = omp_get_wtime();
	if (method == SEQUENTIAL) { gemmSequential<T>(A, B, C); } 
	else if (method == SEQUENTIALTRANSPOSE) { gemmSequentialTranspose<T>(A, B, C); }
	else if (method == OPENMP) { gemmOpenmp<T>(A, B, C); }
	else if (method == PARTITIONEDOPENMP) { gemmPartitionedOpenmp<T>(A, B, C, 128); }
	double endTime = omp_get_wtime();

	if (logLevel >= 3) {
		std::cout << " > [LOGGING] Matrix Multiplication" << std::endl;A.print();
		std::cout << " > [LOGGING] x " << std::endl;B.print();
		std::cout << " > [LOGGING] = " << std::endl;C.print();
	}

	if (logLevel >= 2) {
		std::cout << " = [TIMER] Matrix Multiplication completed in " << endTime - startTime << std::endl;
	}
}

int main(int argc, const char * argv[]) {
	MatrixFlatten<int> matA(1024, 1024);
	MatrixFlatten<int> matB(1024, 1024);
	MatrixFlatten<int> answer(1024, 1024);
	MatrixFlatten<int> result(1024, 1024);

	matA.fillRandomValue(-5, 5);
	matB.fillRandomValue(-10, 10);
	gemm<int>(matA, matB, result, SEQUENTIAL, 2);
	gemm<int>(matA, matB, result, OPENMP, 2);
	// still not correct
	gemm<int>(matA, matB, result, SEQUENTIALTRANSPOSE, 2);
	gemm<int>(matA, matB, answer, PARTITIONEDOPENMP, 2);

	for (int i=0;i<matA.getRow();i++)
		for (int j=0;j<matB.getCol();j++) 
			assert(result(i, j) == answer(i,j));

}
