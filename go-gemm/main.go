package main
import (
	"fmt"
	"log"
	"time"
	"flag"
	"go-gemm/matmult"
)

func main() {

	var M, N, K int
	var matMultType string

	flag.IntVar(&M, "M", 1024, "N row of matrix A")
	flag.IntVar(&N, "N", 1024, "N col of matrix A")
	flag.IntVar(&K, "K", 1024, "N col of resulting matrix")
	flag.StringVar(&matMultType, "matmult", "matMult", "Alg to test")

	flag.Parse()

	matA := make([]int, M*N)
	matB := make([]int, N*K)

	matmult.InitMatrix(&matA, M, N)
	matmult.InitMatrix(&matB, N, K)

	runner, err := matmult.GetMatrixMultiplicationAlg(matMultType); if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	_ = runner(&matA, &matB, M, N, K)
	end := time.Now()
	fmt.Println("matrix mult run in ", end.Sub(start))
}
