package NeuralNetwork

type network struct {
	layers      []layer
	correctness float64
}

func (myNetwork *network) initializeNetwork(numberOfLayers int, nodesPerLayer []int) {
	myNetwork.layers = make([]layer, numberOfLayers)
	for i := range myNetwork.layers {
		if i == len(myNetwork.layers)-1 {
			myNetwork.layers[i].initializeLayer(numberOfLayers, nodesPerLayer[i], 0)
		}
		myNetwork.layers[i].initializeLayer(numberOfLayers, nodesPerLayer[i], nodesPerLayer[i-1])
	}
}
