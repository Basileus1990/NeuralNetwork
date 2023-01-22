package NeuralNetwork

type layer struct {
	nodes []node
}

func (myLayer *layer) initializeLayer(numberOfLayers int, numberOfNodes int, prevLayer *layer, nextLayer *layer) {
	myLayer.nodes = make([]node, numberOfNodes)
	for i := range myLayer.nodes {
		myLayer.nodes[i].initializeNode(prevLayer, nextLayer)
	}
}
