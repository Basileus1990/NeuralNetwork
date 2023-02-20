package network

import (
	"math"
	"sync"
)

type Network struct {
	layers       []layer
	cost         float64
	outputLabels []string
}

func (net *Network) InitializeNetwork(nodesPerLayer []int, outputLabels []string) {
	net.outputLabels = outputLabels

	net.layers = make([]layer, len(nodesPerLayer))
	if len(net.layers) == 1 {
		net.layers[0].initializeLayer(nodesPerLayer[0], 0, nil, nil)
		return
	}

	net.layers[0].initializeLayer(nodesPerLayer[0], nodesPerLayer[1], nil, &net.layers[1])
	for i := 1; i < len(net.layers)-1; i++ {
		net.layers[i].initializeLayer(nodesPerLayer[i], nodesPerLayer[i+1], &net.layers[i-1], &net.layers[i+1])
	}
	net.layers[len(net.layers)-1].initializeLayer(nodesPerLayer[len(net.layers)-1], 0, &net.layers[len(net.layers)-2], nil)
}

func (net *Network) InitializeEmptyNetwork(nodesPerLayer []int, outputLabels []string) {
	net.outputLabels = outputLabels
	net.layers = make([]layer, len(nodesPerLayer))
	if len(net.layers) == 1 {
		net.layers[0].initializeEmptyLayer(nodesPerLayer[0], 0, nil, nil)
		return
	}

	net.layers[0].initializeEmptyLayer(nodesPerLayer[0], nodesPerLayer[1], nil, &net.layers[1])
	for i := 1; i < len(net.layers)-1; i++ {
		net.layers[i].initializeEmptyLayer(nodesPerLayer[i], nodesPerLayer[i+1], &net.layers[i-1], &net.layers[i+1])
	}
	net.layers[len(net.layers)-1].initializeEmptyLayer(nodesPerLayer[len(net.layers)-1], 0, &net.layers[len(net.layers)-2], nil)
}

// returns the structure of the network - number of layers and nodes per layer
func (net *Network) GetNetworkStructure() []int {
	nodesPerLayer := make([]int, len(net.layers))
	for i := range nodesPerLayer {
		nodesPerLayer[i] = len(net.layers[i].nodes)
	}
	return nodesPerLayer
}

// returns network's output labels
func (net *Network) GetOutputLabels() []string {
	return net.outputLabels
}

// for given input it calculates the values of output nodes
// after this function these nodes' values are ready to be exratced
func (net *Network) calculateOutput(inputData []float64) {
	// set input data to the input nodes
	for i := range net.layers[0].nodes {
		net.layers[0].nodes[i].value = inputData[i]
	}

	// only calculates the next layer for first layer as it shouldn't be sigmoidized
	net.layers[0].calculateNextLayer()
	for i := 1; i < len(net.layers)-1; i++ {
		net.layers[i].sigmoidizeNodes()
		net.layers[i].calculateNextLayer()
	}
	// only sigmoidized the last layer as there are no next layers
	net.layers[len(net.layers)-1].sigmoidizeNodes()
}

// calculates network's average cost for given data sets
func (net *Network) CalculateCost(lock *sync.Mutex, dataSets DataSets) {
	combinedCost := 0.0
	for i := range dataSets {
		data := dataSets.GetSafeDataSetCopy(lock, i)
		for key, value := range net.GetOutputsMap(data.inputs) {
			if key == data.expectedOutput {
				combinedCost += math.Pow(1-value, 2) //FIXME: make it usable for something other than simgoid
			} else {
				combinedCost += math.Pow(value, 2)
			}
		}
	}

	net.cost = combinedCost / float64(len(dataSets))
}

func (net *Network) GetCost() float64 {
	return net.cost
}

// Calculates the output and returns a map where for each output node its lalbel
// is the key and value is the map value
func (net *Network) GetOutputsMap(inputData []float64) map[string]float64 {
	net.calculateOutput(inputData)

	resultMap := make(map[string]float64)
	for i, node := range net.layers[len(net.layers)-1].nodes {
		resultMap[net.outputLabels[i]] = node.value
	}
	return resultMap
}

// Calculates the output and returns a label of the best output node and its value
func (net *Network) GetBestOutput(inputData []float64) (string, float64) {
	net.calculateOutput(inputData)

	bestValue := 0.0
	labelIndex := 0
	for i, node := range net.layers[len(net.layers)-1].nodes {
		if node.value > bestValue {
			bestValue = node.value
			labelIndex = i
		}
	}

	return net.outputLabels[labelIndex], bestValue
}

// returns weight and bias of requested node
func (net *Network) GetNodeBias(layerIndex, nodeIndex int) float64 {
	bias := net.layers[layerIndex].nodes[nodeIndex].bias
	return bias
}

// returns weight of requested node
func (net *Network) GetNodeWeight(layerIndex, nodeIndex, weightIndex int) float64 {
	weight := net.layers[layerIndex].nodes[nodeIndex].weights[weightIndex]
	return weight
}

// sets bias of requested node if arguments are correct
func (net *Network) SetNodeBias(layerIndex, nodeIndex int, newBias float64) {
	net.layers[layerIndex].nodes[nodeIndex].bias = newBias
}

// sets weight of requested node's weight if arguments are correct
func (net *Network) SetNodeWeight(layerIndex, nodeIndex, weightIndex int, newWeight float64) {
	net.layers[layerIndex].nodes[nodeIndex].weights[weightIndex] = newWeight
}

// returns a network with the same structure, wieghts and biases
// func (net *Network) CopyNetwork() (copy Network, err error) {
// 	nodesPerLayer := net.GetNetworkStructure()
// 	copy.InitializeNetwork(nodesPerLayer, net.outputLabels)
// 	// iterating over all layers
// 	for i := 0; i < len(nodesPerLayer); i++ {
// 		// iterating over all nodes in a layer
// 		for j := 0; j < nodesPerLayer[i]; j++ {
// 			bias, err := net.GetNodeBias(i, j)
// 			if err != nil {
// 				return copy, err
// 			}
// 			err = copy.SetNodeBias(i, j, bias)
// 			if err != nil {
// 				return copy, err
// 			}

// 			// iteratig over all weights
// 			for k := 0; i != len(nodesPerLayer)-1 && k < nodesPerLayer[i+1]; k++ {
// 				weight, err := net.GetNodeWeight(i, j, k)
// 				if err != nil {
// 					return copy, err
// 				}
// 				err = copy.SetNodeWeight(i, j, k, weight)
// 				if err != nil {
// 					return copy, err
// 				}
// 			}
// 		}
// 	}
// 	return copy, nil
// }
