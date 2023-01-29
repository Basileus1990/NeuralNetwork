package NeuralNetwork

import "sync"

type network struct {
	layers []layer
}

func (net *network) initializeNetwork(nodesPerLayer []int) {
	net.layers = make([]layer, len(nodesPerLayer))
	for i := range net.layers {
		if len(net.layers) == 1 {
			net.layers[i].initializeLayer(nodesPerLayer[i], 0, nil, nil)
			continue
		} else if i == 0 {
			net.layers[i].initializeLayer(nodesPerLayer[i], nodesPerLayer[i+1], nil, &net.layers[i+1])
			continue
		} else if i == len(net.layers)-1 {
			net.layers[i].initializeLayer(nodesPerLayer[i], 0, &net.layers[i-1], nil)
			continue
		}
		net.layers[i].initializeLayer(nodesPerLayer[i], nodesPerLayer[i+1], &net.layers[i-1], &net.layers[i+1])
	}
}

func (net *network) calculateOutput(wg *sync.WaitGroup, inputData []float64) {
	if wg != nil {
		defer wg.Done()
	}
	// set input data to the input nodes
	for i := range net.layers[0].nodes {
		net.layers[0].nodes[i].value = inputData[i]
	}

	for i := 0; i < len(net.layers); i++ {
		// exludes first layer as it shouldn't be sigmoizdized
		if i != 0 {
			net.layers[i].sigmoidizeNodes()
		}
		// exludes last layer as it does't have any next nodes
		if i != len(net.layers)-1 {
			net.layers[i].calculateNextLayer()
		}
	}
}

func (net *network) getOutputValuesSlice() []float64 {
	output := make([]float64, len(net.layers[len(net.layers)-1].nodes))
	for i := range output {
		output[i] = net.layers[len(net.layers)-1].nodes[i].value
	}
	return output
}
