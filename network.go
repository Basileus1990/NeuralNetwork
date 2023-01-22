package NeuralNetwork

type network struct {
	layers []layer
}

func (myNetwork *network) initializeNetwork(numberOfLayers int, nodesPerLayer []int) {
	myNetwork.layers = make([]layer, numberOfLayers)
	for i := range myNetwork.layers {
		if i == 0 {
			myNetwork.layers[i].initializeLayer(numberOfLayers, nodesPerLayer[i], nil, &myNetwork.layers[i+1])
			continue
		} else if i == len(myNetwork.layers)-1 {
			myNetwork.layers[i].initializeLayer(numberOfLayers, nodesPerLayer[i], &myNetwork.layers[i-1], nil)
			continue
		}
		myNetwork.layers[i].initializeLayer(numberOfLayers, nodesPerLayer[i], &myNetwork.layers[i-1], &myNetwork.layers[i+1])
	}
}
