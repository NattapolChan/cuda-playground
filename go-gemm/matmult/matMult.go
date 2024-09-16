package matmult

import (
	"time"
	"sync"
)

func init() {
	// this will not work: function returns before all go routine finish
	Register("matMult", func (matrixA *[]int, matrixB *[]int, M, N, K int) []int {
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
	})

	Register("matMultSync", func (matrixA *[]int, matrixB *[]int, M, N, K int) []int {
		matC := make([]int, M*K)
		var wg sync.WaitGroup
		for i:= 0; i< M;i++ {
			for j := 0;j<K ;j++ {
				wg.Add(1)
				go func (i int, j int) {
					defer wg.Done()
					for it :=0 ;it < N ;it++ {
						matC[i*K+j] += (*matrixA)[i*N+it]  * (*matrixB)[it*K+j]
					}
				} (i, j)
			}
		}
		wg.Wait()
		return matC
	})

	// TO FIX
	Register("matMultPartitioned", func (matrixA *[]int, matrixB *[]int, M, N, K int) []int {
		blockSize := 64
		workerLimit := make(chan struct{}, 16)
		matC := make([]int, M*K)
		var wg sync.WaitGroup

		for i:=0; i<M/blockSize; i++ {
			for j:=0;j<N/blockSize; j++ {
				for k:=0;k<K/blockSize; k++ {
					wg.Add(1)
					workerLimit <- struct{}{}
					go func (i, j, k int) {
						defer func() {
							wg.Done()
							<-workerLimit
						}()
						for bi:=0; bi<blockSize;bi++ {
							for bk:=0;bk<blockSize;bk++ {
									idc := i*blockSize*K + bi*K + k*blockSize + bk
									for it:=0;it<blockSize;it++ {
										ida := i*blockSize*N + bi*N + j*blockSize + it
										idb := j*blockSize*K + it*K + k*blockSize + bk
										matC[idc] += (*matrixA)[ida] * (*matrixB)[idb]
									}
							}
						}
					} (i, j, k)
				}
			}
		}

		wg.Wait()

		return matC
	})
}

