package main

import (
	"fmt"
	//"math/rand"
	"time"
)

var M int
var N int

func main() {
	M = 100
	N = 100
	matA := make([]int, M*N)
	matB := make([]int, M*N)

	initMatrix(&matA)
	initMatrix(&matB)

	start := time.Now()
	matC := matMult(&matA, &matB)
	end := time.Now()
	fmt.Println("matrix mult run in %f", end.Sub(start))

//	printMatrix(&matA)
	fmt.Println()
	fmt.Println()
//	printMatrix(&matB)
	fmt.Println()
	fmt.Println()
	printMatrix(&matC)
}

func initMatrix(matrix *[]int) {
	for i := 0; i < N; i++ {
		for j := 0; j < M; j++ {
			(*matrix)[i*N + j] = 1
		}
	}
}

func printMatrix(matrix *[]int) {
	for i := 0; i < N; i++ {
		for j := 0; j < M; j++ {
			//fmt.Printf("%d ", (*matrix)[i*N + j])
		}
		//fmt.Println()
	}
}

func matMult(matrixA *[]int, matrixB *[]int) []int {
	matC := make([]int, M*N)
	for i:= 0; i< N;i++ {
		for j := 0;j<M ;j++ {
			go func (i int, j int) {
				for it :=0 ;it < N ;it++ {
					matC[i*N+j] += (*matrixA)[i*N+it]  * (*matrixB)[it*N + j]
				}
			} (i, j)
		}
	}
	return matC
}

func matSyncMult(matrixA *[]int, matrixB *[]int) []int {
	matC := make([]int, M*N)
	for i:= 0; i< N;i++ {
		for j := 0;j<M ;j++ {
			go func (i int, j int) {
				for it :=0 ;it < N ;it++ {
					matC[i*N+j] += (*matrixA)[i*N+it]  * (*matrixB)[it*N + j]
				}
			} (i, j)
		}
	}
	return matC
}
