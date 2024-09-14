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

	void print() const {
		for (int i=0;i<numRows;i++) {
			for (int j=0;j<numCols;j++)
				std::cout << (*this) (i, j) << " ";
			std::cout << std::endl;
		}
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
};

template <typename T>
void gemmSequential(const MatrixBase<T>& A, const MatrixBase<T>& B, MatrixBase<T>& C) {
	for (int i = 0; i < A.getRow(); ++i) {
		for (int j = 0; j < B.getCol(); ++j) {
			C(i, j) = 0;
			for (int k = 0; k < A.getCol(); ++k) {
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
	for (int i = 0; i < A.getRow(); ++i) {
		for (int j = 0; j < B.getCol(); ++j) {
			C(i, j) = 0;
			for (int k = 0; k < A.getCol(); ++k) {
				T a = A(i, k);
				T b = B(k, j);
				C(i, j) += a * b;
			}
		}
	}
}

template <typename T>
void gemm(MatrixBase<T>& A, MatrixBase<T>& B, MatrixBase<T>& C, const EnumMatrixMatrixMultiplicationMethod method, int logLevel = 3) {

	assert (A.getCol() == B.getRow() && C.getRow() == A.getRow() && C.getCol() == B.getCol());
	double startTime = omp_get_wtime();
	if (method == SEQUENTIAL) { gemmSequential<T>(A, B, C); } 
	else if (method == SEQUENTIALTRANSPOSE) { gemmSequentialTranspose<T>(A, B, C); }
	else if (method == OPENMP) { gemmOpenmp<T>(A, B, C); }
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
	MatrixFlatten<float> matA(200, 300);
	MatrixFlatten<float> matB(300, 400);
	MatrixFlatten<float> result(200, 400);

	Matrix<float> mA(200, 300);
	Matrix<float> mB(300, 400);
	Matrix<float> res(200, 400);

	matA.fillFixedValue(0.1);
	matB.fillFixedValue(1.2);
	gemm<float>(matA, matB, result, SEQUENTIAL, 2);
	matA.fillFixedValue(0.1);
	matB.fillFixedValue(1.2);
	gemm<float>(matA, matB, result, OPENMP, 2);
	matA.fillFixedValue(0.1);
	matB.fillFixedValue(1.2);
	// still not correct
	gemm<float>(matA, matB, result, SEQUENTIALTRANSPOSE, 2);

	mA.fillFixedValue(0.1);
	mB.fillFixedValue(1.2);
	gemm<float>(mA, mB, res, SEQUENTIAL, 2);
	mA.fillFixedValue(0.1);
	mB.fillFixedValue(1.2);
	gemm<float>(mA, mB, res, OPENMP, 2);
	mA.fillFixedValue(0.1);
	mB.fillFixedValue(1.2);
	gemm<float>(mA, mB, res, SEQUENTIALTRANSPOSE, 2);
}
