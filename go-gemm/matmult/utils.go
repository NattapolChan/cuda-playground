package matmult

import (
	"errors"
	"fmt"
)

var matMultRegistry = make(map[string] func(*[]int, *[]int, int, int, int) []int)

func Register(name string, f func(*[]int, *[]int, int, int, int) []int) {
	matMultRegistry[name] = f
}

func InitMatrix(matrix *[]int, rows int, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			(*matrix)[i*cols + j] = i + j
		}
	}
}

func PrintMatrix(matrix *[]int, rows int, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Printf("%d ", (*matrix)[i*cols + j])
		}
		fmt.Println()
	}
}
func GetMatrixMultiplicationAlg(matMultType string) (func(*[]int, *[]int, int, int, int) []int, error) {

	if matMultFunc, ok := matMultRegistry[matMultType]; ok {
		return matMultFunc, nil
	}

	return nil, errors.New("function not found in registry, check matmult/matMult.go")
}

func GetRegistryList() []string {
	keys := make([]string, 0, len(matMultRegistry))
	for key := range matMultRegistry {
		keys = append(keys, key)
	}
	return keys
}
