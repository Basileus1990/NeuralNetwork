package network

type layer struct {
	nodes []node
}

func (myLayer *layer) initializeLayer(numberOfNodes int, numberOfNextNodes int, prevLayer *layer, nextLayer *layer) {
	myLayer.nodes = make([]node, numberOfNodes)
	for i := range myLayer.nodes {
		myLayer.nodes[i].initializeNode(prevLayer, nextLayer, numberOfNextNodes)
	}
}

func (myLayer *layer) initializeEmptyLayer(numberOfNodes int, numberOfNextNodes int, prevLayer *layer, nextLayer *layer) {
	myLayer.nodes = make([]node, numberOfNodes)
	for i := range myLayer.nodes {
		myLayer.nodes[i].initializeEmptyNode(prevLayer, nextLayer, numberOfNextNodes)
	}
}

func (myLayer *layer) calculateNextLayer() {
	for i := range myLayer.nodes {
		myLayer.nodes[i].calculateNextNodes()
	}
	for i := range myLayer.nodes {
		myLayer.nodes[i].addBias()
	}
}

// uses on all nodes the sigmoid funcion
func (myLayer *layer) sigmoidizeNodes() {
	for i := range myLayer.nodes {
		myLayer.nodes[i].sigmoidize()
	}
}
