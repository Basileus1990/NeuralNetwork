package network

import (
	"math"
	"math/rand"
)

const maxInitialRandomValue = 5

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

	myNode.bias = float64(rand.Intn(maxInitialRandomValue*2+1)*1000)/1000 - maxInitialRandomValue

	if nextLayer == nil {
		return
	}
	myNode.weights = make([]float64, numberOfNextNodes)
	for i := range myNode.weights {
		myNode.weights[i] = float64(rand.Intn(maxInitialRandomValue*2+1)*1000)/1000 - maxInitialRandomValue
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
