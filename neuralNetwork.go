package NeuralNetwork

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

// Contains all networks created to the task
type neuralNetwork struct {
	networks     []network.Network
	outputLabels []string
}

/*
parameters:
-> numberOfTrainingNetworks >= 1
-> nodesPerLayer[i] > 0
-> len(nodesPerLayer) > 0
-> len(outputLabels) == len(nodesPerLayer[output])
each outputLabels string corresponds to one node from the starting from the top

Retrun: an initialized networks ready to be given data and to be trained
*/
func NewNeuralNetwork(numberOfTrainingNetworks int, nodesPerLayer []int, outputLabels []string) (*neuralNetwork, error) {
	// A seed for a random initial network state
	rand.Seed(time.Now().UnixNano())

	if err := validateNetworkInitArguments(numberOfTrainingNetworks, nodesPerLayer, outputLabels); err != nil {
		return nil, err
	}

	// initialize networks
	var myNeuralNetwork neuralNetwork
	myNeuralNetwork.networks = make([]network.Network, numberOfTrainingNetworks)
	myNeuralNetwork.outputLabels = outputLabels
	for i := range myNeuralNetwork.networks {
		myNeuralNetwork.networks[i].InitializeNetwork(nodesPerLayer)
	}
	return &myNeuralNetwork, nil
}

// If network is not initialized then informs the user about it
func (myNeuralNetwork *neuralNetwork) PrintNetworkSchema() {
	_, nodesPerLayer := myNeuralNetwork.networks[0].NetworkStructure()

	fmt.Println("<========================>")
	fmt.Println(" A neural network schema:")
	for i, nodes := range nodesPerLayer {
		fmt.Printf(" Layer %d: %d nodes\n", i, nodes)
	}
	fmt.Println("<========================>")
}

/*
Parameters:
-> inputData: len(inputData) == GetLenOfInNodes()
0 <= inputData[i] <= 1
-> outputLabels: len(outputLabes) == GetLenOfOutNodes()
each outputLabels string corresponds to one node from the starting from the top

Returns:
-> A map containing all output layer's node values with the key given by the user
*/
// TODO:
func (myNeuralNetwork *neuralNetwork) GetOutputMap(inputData []float64) (map[string]float64, error) {
	err := myNeuralNetwork.calculateOutputs(inputData)
	if err != nil {
		return nil, err
	}
	resultMap := make(map[string]float64)
	resultSlice := myNeuralNetwork.networks[0].GetOutputValuesSlice() // FIXME:
	for i := range resultSlice {
		resultMap[myNeuralNetwork.outputLabels[i]] = resultSlice[i]
	}
	return resultMap, nil
}

/*
Parameters:
-> inputData: len(inputData) == GetLenOfInNodes()
0 <= inputData[] <= 1
-> outputLabels: len(outputLabes) == GetLenOfOutNodes()
each outputLabels string corresponds to one node from the starting from the top

Returns:
-> The best output node value and it's key
*/
// TODO:
func (myNeuralNetwork *neuralNetwork) GetBestOutput(inputData []float64) (string, float64) {
	return "", 0
}

func (myNeuralNetwork *neuralNetwork) GetLenOfInNodes() int {
	_, nodes := myNeuralNetwork.networks[0].NetworkStructure()
	return nodes[0]
}

func (myNeuralNetwork *neuralNetwork) GetLenOfOutNodes() int {
	numberOfLayers, nodes := myNeuralNetwork.networks[0].NetworkStructure()
	return nodes[numberOfLayers-1]
}

func (myNeuralNetwork *neuralNetwork) calculateOutputs(inputData []float64) error {
	err := myNeuralNetwork.validateInputData(inputData)
	if err != nil {
		return err
	}

	const concurrentThershold = 5
	if len(myNeuralNetwork.networks) > concurrentThershold {
		var wg sync.WaitGroup
		for i := range myNeuralNetwork.networks {
			wg.Add(1)
			go myNeuralNetwork.networks[i].CalculateOutput(&wg, inputData)
		}

		wg.Wait()
	} else {
		for i := range myNeuralNetwork.networks {
			myNeuralNetwork.networks[i].CalculateOutput(nil, inputData)
		}
	}
	return nil
}

func (myNeuralNetwork *neuralNetwork) validateInputData(inputData []float64) error {
	if len(inputData) != myNeuralNetwork.GetLenOfInNodes() {
		return errors.New("number of input data has to be the same as number of input nodes")
	}
	for _, v := range inputData {
		if v < 0 || v > 1 {
			return errors.New("network input has to be beetween [0,1]")
		}
	}
	return nil
}

func validateNetworkInitArguments(numberOfTrainingNetworks int, nodesPerLayer []int, outputLabels []string) error {
	if numberOfTrainingNetworks <= 0 {
		return errors.New("number of training networks has to bigger than 0")
	}
	if len(nodesPerLayer) <= 0 {
		return errors.New("number of layers has to be bigger than 0")
	}
	for _, v := range nodesPerLayer {
		if v <= 0 {
			return errors.New("number of nodes per layer can't be lower than 1")
		}
	}
	if len(outputLabels) != nodesPerLayer[len(nodesPerLayer)-1] {
		return errors.New("number of output labes has to be the same as number of output nodes")
	}

	return nil
}
