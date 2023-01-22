package NeuralNetwork

import (
	"errors"
	"fmt"
)

// Contains all networks created to the task
type neuralNetwork struct {
	networks []network
}

/*
require:

	numberOfTrainingNetworks >= 1
	numberOfLayers > 0
	nodesPerLayer[i] > 0
	len(nodesPerLayer) == numberOfLayers

effect:

	returns initialized networks ready to be given data and to be trained
*/
func NewNeuralNetwork(numberOfTrainingNetworks int, numberOfLayers int, nodesPerLayer []int) (*neuralNetwork, error) {
	if err := validateNetworkInitArguments(numberOfTrainingNetworks, numberOfLayers, nodesPerLayer); err != nil {
		return nil, err
	}

	// initialize networks
	var myNeuralNetwork neuralNetwork
	myNeuralNetwork.networks = make([]network, numberOfTrainingNetworks)
	for i := range myNeuralNetwork.networks {
		myNeuralNetwork.networks[i].initializeNetwork(numberOfLayers, nodesPerLayer)
	}
	return &myNeuralNetwork, nil
}

// If network is not initialized then informs the user about it
func (myNeuralNetwork *neuralNetwork) PrintNetworkSchema() {
	if myNeuralNetwork == nil {
		fmt.Println("Network hasn't been initialized")
		return
	}

	printedNetwork := myNeuralNetwork.networks[0]

	fmt.Println("<========================>")
	fmt.Println(" A neural network schema:")
	for i, myLayer := range printedNetwork.layers {
		fmt.Printf(" Layer %d: %d nodes\n", i, len(myLayer.nodes))
	}
	fmt.Println("<========================>")
}

func validateNetworkInitArguments(numberOfTrainingNetworks int, numberOfLayers int, nodesPerLayer []int) error {
	if numberOfTrainingNetworks <= 0 {
		return errors.New("number of training networks has to bigger than 0")
	}
	if numberOfLayers <= 0 {
		return errors.New("number of layers has to bigger than 0")
	}
	if len(nodesPerLayer) != numberOfLayers {
		return errors.New("the specified number of nodes per layer has to be same as the number of layers")
	}
	for _, v := range nodesPerLayer {
		if v <= 0 {
			return errors.New("number of nodes can't be lower than 1")
		}
	}

	return nil
}
