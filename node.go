package NeuralNetwork

import "math/rand"

type node struct {
	value     float64
	bias      float64
	prevLayer *layer
	nextLayer *layer
	weights   []float64 // weights for the next nodes
}

// gives the node random weights and biases
func (myNode *node) initializeNode(prevLayer *layer, nextLayer *layer) {
	myNode.prevLayer = prevLayer
	myNode.nextLayer = nextLayer

	myNode.bias = float64(rand.Intn(101)) / 100
	if myNode.nextLayer == nil {
		return
	}

	myNode.weights = make([]float64, len((*nextLayer).nodes))
	for i := range myNode.weights {
		myNode.weights[i] = float64(rand.Intn(101)) / 100
	}
}
