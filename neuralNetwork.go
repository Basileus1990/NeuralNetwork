package NeuralNetwork

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

// Contains all networks created to the task
type neuralNetwork struct {
	networks []network
}

/*
require:

	numberOfTrainingNetworks >= 1
	nodesPerLayer[i] > 0
	len(nodesPerLayer) > 0

effect:

	returns initialized networks ready to be given data and to be trained
*/
func NewNeuralNetwork(numberOfTrainingNetworks int, nodesPerLayer []int) (*neuralNetwork, error) {
	// A seed for a random initial network state
	rand.Seed(time.Now().UnixNano())

	if err := validateNetworkInitArguments(numberOfTrainingNetworks, nodesPerLayer); err != nil {
		return nil, err
	}

	// initialize networks
	var myNeuralNetwork neuralNetwork
	myNeuralNetwork.networks = make([]network, numberOfTrainingNetworks)
	for i := range myNeuralNetwork.networks {
		myNeuralNetwork.networks[i].initializeNetwork(nodesPerLayer)
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

func (myNeuralNetwork *neuralNetwork) GetLenOfInNodes() int {
	return len(myNeuralNetwork.networks[0].layers[0].nodes)
}

func (myNeuralNetwork *neuralNetwork) GetLenOfOutNodes() int {
	return len(myNeuralNetwork.networks[0].layers[len(myNeuralNetwork.networks[0].layers)-1].nodes)
}

func (myNeuralNetwork *neuralNetwork) CalculateOutput() []float64 {
	return []float64{1.0}
}

func validateNetworkInitArguments(numberOfTrainingNetworks int, nodesPerLayer []int) error {
	if numberOfTrainingNetworks <= 0 {
		return errors.New("number of training networks has to bigger than 0")
	}
	if len(nodesPerLayer) <= 0 {
		return errors.New("number of layers has to bi bigger than 0")
	}
	for _, v := range nodesPerLayer {
		if v <= 0 {
			return errors.New("number of nodes per layer can't be lower than 1")
		}
	}

	return nil
}
