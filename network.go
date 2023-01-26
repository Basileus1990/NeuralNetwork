package NeuralNetwork

type network struct {
	layers []layer
}

func (myNetwork *network) initializeNetwork(nodesPerLayer []int) {
	myNetwork.layers = make([]layer, len(nodesPerLayer))
	for i := range myNetwork.layers {
		if len(myNetwork.layers) == 1 {
			myNetwork.layers[i].initializeLayer(nodesPerLayer[i], nil, nil)
			continue
		}
		if i == 0 {
			myNetwork.layers[i].initializeLayer(nodesPerLayer[i], nil, &myNetwork.layers[i+1])
			continue
		} else if i == len(myNetwork.layers)-1 {
			myNetwork.layers[i].initializeLayer(nodesPerLayer[i], &myNetwork.layers[i-1], nil)
			continue
		}
		myNetwork.layers[i].initializeLayer(nodesPerLayer[i], &myNetwork.layers[i-1], &myNetwork.layers[i+1])
	}
}
