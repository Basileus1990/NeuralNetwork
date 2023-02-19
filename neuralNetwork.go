package NeuralNetwork

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Basileus1990/NeuralNetwork.git/integral/network"
)

// for random generation of mutations and random initialization
func init() {
	rand.Seed(time.Now().UnixNano())
}

type neuralNetwork struct {
	network                  network.Network
	numberOfTrainingNetworks int
	trainingDataSets         []network.DataSet
}

// Retruns an initialized neural network ready to be given data and to be trained.
// Number of training networks has to be bigger than 0.
// Amount of layers and nodes has to bigger than 0.
// amount of output labels has to be equal to number of output nodes
func NewNeuralNetwork(numberOfTrainingNetworks int, nodesPerLayer []int, outputLabels []string) (*neuralNetwork, error) {
	if err := validateNetworkInit(numberOfTrainingNetworks, nodesPerLayer, outputLabels); err != nil {
		return nil, err
	}

	// initialize networks
	var neuralNet neuralNetwork
	neuralNet.network.InitializeNetwork(nodesPerLayer, outputLabels)
	return &neuralNet, nil
}

func (neuralNet *neuralNetwork) PrintNetworkSchema() {
	nodesPerLayer := neuralNet.network.GetNetworkStructure()

	fmt.Println("<========================>")
	fmt.Println(" A neural network schema:")
	for i, nodes := range nodesPerLayer {
		fmt.Printf(" Layer %d: %d nodes\n", i, nodes)
	}
	fmt.Println("<========================>")
}

// Returns the best a map where output label are keys and outputs are values for given input data.
// Inputs have to be between 0 and 1
func (neuralNet *neuralNetwork) GetOutputMap(inputData []float64) (map[string]float64, error) {
	err := neuralNet.validateInputData(inputData)
	if err != nil {
		return nil, err
	}

	return neuralNet.network.GetOutputsMap(inputData), nil
}

// Returns the best output label for given input data. Inputs have to be between 0 and 1
func (neuralNet *neuralNetwork) GetNetworkResult(inputData []float64) (string, error) {
	err := neuralNet.validateInputData(inputData)
	if err != nil {
		return "", err
	}

	label, _ := neuralNet.network.GetBestOutput(inputData)
	return label, nil
}

// Returns the amount of network's input nodes
func (neuralNet *neuralNetwork) NumberOfInputNodes() int {
	nodesPerLayer := neuralNet.network.GetNetworkStructure()
	return nodesPerLayer[0]
}

// Returns the amount of network's output nodes
func (neuralNet *neuralNetwork) NumberOfOutputNodes() int {
	nodesPerLayer := neuralNet.network.GetNetworkStructure()
	return nodesPerLayer[len(nodesPerLayer)-1]
}

// Assings the given data to the trainer replacing the old data
func (neuralNet *neuralNetwork) LoadTrainingData(inputs [][]float64, outputs []string) error {
	if err := neuralNet.validateTrainingInputData(inputs, outputs); err != nil {
		return err
	}

	dataSets := make([]network.DataSet, len(inputs))
	for i := 0; i < len(dataSets); i++ {
		dataSets[i].SetDataSet(inputs[i], outputs[i])
	}

	neuralNet.trainingDataSets = dataSets
	return nil
}

// Appends given given data set to training data sets
func (neuralNet *neuralNetwork) AddSingleTrainingData(input []float64, output string) error {
	if err := neuralNet.validateTrainingInputData([][]float64{input}, []string{output}); err != nil {
		return err
	}

	var newDataSet network.DataSet
	newDataSet.SetDataSet(input, output)
	neuralNet.trainingDataSets = append(neuralNet.trainingDataSets, newDataSet)
	return nil
}
