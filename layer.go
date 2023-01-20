package NeuralNetwork

type layer struct {
	nodes []node
}

func (myLayer *layer) initializeLayer(numberOfLayers int, numberOfNodes int, numberOfFollowingNodes int) {
	myLayer.nodes = make([]node, numberOfNodes)
	for i := range myLayer.nodes {
		myLayer.nodes[i].initializeNode(numberOfFollowingNodes)
	}
}
