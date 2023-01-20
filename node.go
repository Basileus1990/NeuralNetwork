package NeuralNetwork

import "math/rand"

type node struct {
	value   float64
	bias    float64
	weights []float64 // weights for the next nodes
}

// gives the node random weights and biases
func (myNode *node) initializeNode(numberOfPreviousNodes int) {
	myNode.bias = float64(rand.Intn(101)) / 100
	if numberOfPreviousNodes == 0 {
		return
	}

	myNode.weights = make([]float64, numberOfPreviousNodes)
	for i := range myNode.weights {
		myNode.weights[i] = float64(rand.Intn(101)) / 100
	}
}
