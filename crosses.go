package regression

import (
	"math"
	"strconv"
)

type featureCross interface {
	Calculate([]float64) []float64 //must return the same number of features each run
	ExtendNames(map[int]string, int) int
}

type functionalCross struct {
	functionName string
	boundVars    []int
	crossFn      func([]float64) []float64
}

func (c *functionalCross) Calculate(input []float64) []float64 {
	return c.crossFn(input)
}

func (c *functionalCross) ExtendNames(input map[int]string, initialSize int) int {
	for i, varIndex := range c.boundVars {
		if input[varIndex] != "" {
			input[initialSize+i] = "(" + input[varIndex] + ")" + c.functionName
		}
	}
	return len(c.boundVars)
}

// Feature cross based on computing the log of an input.
func LogCross(i int) featureCross {
	return &functionalCross{
		functionName: "log",
		boundVars:    []int{i},
		crossFn: func(vars []float64) []float64 {

			return []float64{math.Log(vars[i])}
		},
	}
}

// Feature cross based on the multiplication of multiple inputs.
func MultiplierCross(vars ...int) featureCross {
	name := ""
	for i, v := range vars {
		name += strconv.Itoa(v)
		if i < (len(vars) - 1) {
			name += "*"
		}
	}

	return &functionalCross{
		functionName: name,
		boundVars:    vars,
		crossFn: func(input []float64) []float64 {
			var output float64 = 1
			for _, variableIndex := range vars {
				output *= input[variableIndex]
			}
			return []float64{output}
		},
	}
}
