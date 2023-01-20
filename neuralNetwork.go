package NeuralNetwork

import (
	"errors"
	"fmt"
)

var networks []network = nil

/*
require:

	numberOfTrainingNetworks >= 1
	numberOfLayers > 0
	nodesPerLayer[i] > 0
	len(nodesPerLayer) == numberOfLayers

effect:

	an initialized networks ready to be given data and to be trained
*/
func CreateNetwork(numberOfTrainingNetworks int, numberOfLayers int, nodesPerLayer []int) error {
	if err := validateNetworkInitArguments(numberOfTrainingNetworks, numberOfLayers, nodesPerLayer); err != nil {
		return err
	}

	// initialize networks
	networks = make([]network, numberOfTrainingNetworks)
	for i := range networks {
		networks[i].initializeNetwork(numberOfLayers, nodesPerLayer)
	}
	return nil
}

// If network is not initialized then informs the user about it
func PrintNetworkSchema() {
	if networks == nil {
		fmt.Println("Network hasn't been initialized")
		return
	}
}

func validateNetworkInitArguments(numberOfTrainingNetworks int, numberOfLayers int, nodesPerLayer []int) error {
	if numberOfTrainingNetworks <= 0 {
		return errors.New("number of training networks has to bigger than 0")
	}
	if numberOfLayers <= 0 {
		return errors.New("number of layers has to bigger than 0")
	}
	if len(nodesPerLayer) == numberOfLayers {
		return errors.New("the specified number of nodes per layer has to be same as the number of layers")
	}
	for _, v := range nodesPerLayer {
		if v <= 0 {
			return errors.New("number of nodes can't be lower than 1")
		}
	}

	return nil
}
