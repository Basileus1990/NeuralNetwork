package network

import (
	"math"
	"math/rand"
)

type node struct {
	value     float64
	bias      float64
	prevLayer *layer
	nextLayer *layer
	weights   []float64 // weights for the next nodes
}

// gives the node random weights and biases
func (myNode *node) initializeNode(prevLayer *layer, nextLayer *layer, numberOfNextNodes int) {
	myNode.prevLayer = prevLayer
	myNode.nextLayer = nextLayer

	myNode.bias = float64(rand.Intn(20001))/10000 - 1

	if nextLayer == nil {
		return
	}
	myNode.weights = make([]float64, numberOfNextNodes)
	for i := range myNode.weights {
		myNode.weights[i] = float64(rand.Intn(20001))/10000 - 1
	}
}

func (myNode *node) calculateNextNodes() {
	for i := range myNode.nextLayer.nodes {
		addedValue := myNode.value*myNode.weights[i] + myNode.bias
		myNode.nextLayer.nodes[i].value += addedValue
	}
}

func (myNode *node) sigmoidize() {
	myNode.value = 1.0 / (1 + math.Exp(myNode.value))
}
