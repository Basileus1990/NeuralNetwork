package network

import (
	"errors"
	"sync"
)

type Network struct {
	layers []layer
}

func (net *Network) InitializeNetwork(nodesPerLayer []int) {
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

// returns the structure of the network - number of layers and nodes per layer
func (net *Network) NetworkStructure() (numberOfLayers int, nodesPerLayer []int) {
	numberOfLayers = len(net.layers)
	nodesPerLayer = make([]int, numberOfLayers)
	for i := range nodesPerLayer {
		nodesPerLayer[i] = len(net.layers[i].nodes)
	}
	return len(net.layers), nodesPerLayer
}

func (net *Network) CalculateOutput(wg *sync.WaitGroup, inputData []float64) {
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

func (net *Network) GetOutputValuesSlice() (output []float64) {
	output = make([]float64, len(net.layers[len(net.layers)-1].nodes))
	for i := range output {
		output[i] = net.layers[len(net.layers)-1].nodes[i].value
	}
	return output
}

// returns weight and bias of requested node
func (net *Network) GetNodeBias(layerIndex, nodeIndex int) (bias float64, err error) {
	layers, nodesPerLayer := net.NetworkStructure()
	if layerIndex < 0 || layerIndex >= layers {
		return 0, errors.New("layer index out of range")
	}
	if nodeIndex < 0 || nodeIndex >= nodesPerLayer[layerIndex] {
		return 0, errors.New("node index out of range")
	}

	bias = net.layers[layerIndex].nodes[nodeIndex].bias
	return bias, nil
}

// returns weight of requested node
func (net *Network) GetNodeWeight(layerIndex, nodeIndex, weightIndex int) (weight float64, err error) {
	layers, nodesPerLayer := net.NetworkStructure()
	if layerIndex < 0 || layerIndex >= layers {
		return 0, errors.New("layer index out of range")
	}
	if nodeIndex < 0 || nodeIndex >= nodesPerLayer[layerIndex] {
		return 0, errors.New("node index out of range")
	}
	// exludes last layer because it doesn't have any child nodes
	if layerIndex == len(nodesPerLayer)-1 {
		return 0, errors.New("there are no weights in output layer")
	}
	if weightIndex < 0 || weightIndex >= nodesPerLayer[layerIndex+1] {
		return 0, errors.New("weight index out of range")
	}

	weight = net.layers[layerIndex].nodes[nodeIndex].weights[weightIndex]
	return weight, nil
}

// sets bias of requested node if arguments are correct
func (net *Network) SetNodeBias(layerIndex, nodeIndex int, newBias float64) (err error) {
	layers, nodesPerLayer := net.NetworkStructure()
	if layerIndex < 0 || layerIndex >= layers {
		return errors.New("layer index out of range")
	}
	if nodeIndex < 0 || nodeIndex >= nodesPerLayer[layerIndex] {
		return errors.New("node index out of range")
	}

	net.layers[layerIndex].nodes[nodeIndex].bias = newBias
	return nil
}

func (net *Network) SetNodeWeight(layerIndex, nodeIndex, weightIndex int, newWeight float64) (err error) {
	layers, nodesPerLayer := net.NetworkStructure()
	if layerIndex < 0 || layerIndex >= layers {
		return errors.New("layer index out of range")
	}
	if nodeIndex < 0 || nodeIndex >= nodesPerLayer[layerIndex] {
		return errors.New("node index out of range")
	}
	// exludes last layer because it doesn't have any child nodes
	if layerIndex == len(nodesPerLayer)-1 {
		return errors.New("there are no weights in output layer")
	}
	if weightIndex < 0 || weightIndex >= nodesPerLayer[layerIndex+1] {
		return errors.New("weight index out of range")
	}

	net.layers[layerIndex].nodes[nodeIndex].weights[weightIndex] = newWeight
	return nil
}

// returns a network with the same structure, wieghts and biases
func (net *Network) CopyNetwork() (copy Network, err error) {
	layers, nodesPerLayer := net.NetworkStructure()
	copy.InitializeNetwork(nodesPerLayer)
	// iterating over all layers
	for i := 0; i < layers; i++ {
		// iterating over all nodes in a layer
		for j := 0; j < nodesPerLayer[i]; j++ {
			bias, err := net.GetNodeBias(i, j)
			if err != nil {
				return copy, err
			}
			err = copy.SetNodeBias(i, j, bias)
			if err != nil {
				return copy, err
			}

			// iteratig over all weights
			for k := 0; i != layers-1 && k < nodesPerLayer[i+1]; k++ {
				weight, err := net.GetNodeWeight(i, j, k)
				if err != nil {
					return copy, err
				}
				err = copy.SetNodeWeight(i, j, k, weight)
				if err != nil {
					return copy, err
				}
			}
		}
	}
	return copy, nil
}
