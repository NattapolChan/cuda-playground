package main

import (
	"fmt"
	"sync"
	"time"
)

var M int
var N int
var K int

func main() {
	M = 1024
	N = 1024
	K = 1024
	matA := make([]int, M*N)
	matB := make([]int, N*K)

	initMatrix(&matA, M, N)
	initMatrix(&matB, N, K)

	start := time.Now()
	_ = matMult(&matA, &matB)
	end := time.Now()
	fmt.Println("matrix mult run in %f", end.Sub(start))

	// fmt.Println()
	// fmt.Println()
	// printMatrix(&matC) // this will be incorrect

	start = time.Now()
	_ = matSyncMult(&matA, &matB)
	end = time.Now()
	fmt.Println("matrix mult with sync run in %f", end.Sub(start))

	start = time.Now()
	_ = matMultPartitioned(&matA, &matB)
	end = time.Now()
	fmt.Println("matrix mult with sync run in %f", end.Sub(start))

	// printMatrix(&matA, M, N)
	// fmt.Println()
	// printMatrix(&matB, N, K)

	// printMatrix(&matC, M, K)
	fmt.Println()
	// printMatrix(&matD, M, K)
}

func initMatrix(matrix *[]int, rows int, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			(*matrix)[i*cols + j] = i + j
		}
	}
}

func printMatrix(matrix *[]int, rows int, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Printf("%d ", (*matrix)[i*cols + j])
		}
		fmt.Println()
	}
}


// this will not work: function returns before all go routine finish
func matMult(matrixA *[]int, matrixB *[]int) []int {
	matC := make([]int, M*K)
	for i:= 0; i< M;i++ {
		for j := 0;j<K ;j++ {
			go func (i int, j int) {
				for it :=0 ;it < N ;it++ {

					// to test sync
					if it == N-1 {
						time.Sleep(2 * time.Second)
					}

					matC[i*K+j] += (*matrixA)[i*N+it]  * (*matrixB)[it*K + j]
				}
			} (i, j)
		}
	}
	return matC
}

func matSyncMult(matrixA *[]int, matrixB *[]int) []int {
	matC := make([]int, M*K)
	var wg sync.WaitGroup
	
	for i:= 0; i< M;i++ {
		for j := 0;j<K ;j++ {
			wg.Add(1)
			go func (i int, j int) {
				defer wg.Done()
				for it :=0 ;it < N ;it++ {
					// to test
					// if it == N-3 {
						// time.Sleep(2 * time.Second)
					// }
					matC[i*K+j] += (*matrixA)[i*N+it]  * (*matrixB)[it*K+j]
				}
			} (i, j)
		}
	}

	wg.Wait()
	return matC
}

// TO FIX
func matMultPartitioned(matrixA *[]int, matrixB *[]int) []int {
	blockSize := 128
	workerLimit := make(chan struct{}, 32)
	matC := make([]int, M*K)
	var wg sync.WaitGroup

	for i:=0; i<M/blockSize; i++ {
		for j:=0;j<N/blockSize; j++ {
			for k:=0;k<K/blockSize; k++ {
				wg.Add(blockSize * blockSize)
				for bi:=0; bi<blockSize;bi++ {
					for bk:=0;bk<blockSize;bk++ {
						workerLimit <- struct{}{}
						go func (i int, j int, k int, bi int, bk int) {
							defer func() {
								wg.Done()
								<-workerLimit
							}()
							idc := i*blockSize*K + bi*K + k*blockSize + bk
							for it:=0;it<blockSize;it++ {
								ida := i*blockSize*N + bi*N + j*blockSize + it
								idb := j*blockSize*K + it*K + k*blockSize + bk
								A := (*matrixA)[ida] 
								B := (*matrixB)[idb]
								matC[idc] +=  A * B 
							}
						} (i, j, k, bi, bk)
					}
				}
			}
		}
	}

	wg.Wait()

	return matC
}
