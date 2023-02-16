package NeuralNetwork

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

// for random generation of mutations and random initialization
func init() {
	rand.Seed(time.Now().UnixNano())
}

// Contains all networks created to the task
type neuralNetwork struct {
	networks     []network.Network
	outputLabels []string
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

// Returns the best a map where output label are keys and outputs are values for given input data.
// Inputs have to be between 0 and 1
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

// Returns the best output label for given input data. Inputs have to be between 0 and 1
// TODO:
func (myNeuralNetwork *neuralNetwork) GetNetworkResult(inputData []float64) string {
	return ""
}

// returns the amount of network's input nodes
func (myNeuralNetwork *neuralNetwork) GetLenOfInNodes() int {
	_, nodes := myNeuralNetwork.networks[0].NetworkStructure()
	return nodes[0]
}

// returns the amount of network's output nodes
func (myNeuralNetwork *neuralNetwork) GetLenOfOutNodes() int {
	numberOfLayers, nodes := myNeuralNetwork.networks[0].NetworkStructure()
	return nodes[numberOfLayers-1]
}

// calculates all networks' outputs concurrently. After this networks will have ready to be extracted outputs
func (myNeuralNetwork *neuralNetwork) calculateOutputs(inputData []float64) error {
	err := myNeuralNetwork.validateInputData(inputData)
	if err != nil {
		return err
	}

	numberOfWorkers := runtime.NumCPU()
	netChan := make(chan *network.Network)
	var wg sync.WaitGroup
	wg.Add(len(myNeuralNetwork.networks))
	for i := 0; i < numberOfWorkers; i++ {
		go func(wg *sync.WaitGroup, netChan chan *network.Network) {
			for net := range netChan {
				net.CalculateOutput(wg, inputData)
			}
		}(&wg, netChan)
	}
	for _, net := range myNeuralNetwork.networks {
		netChan <- &net
	}
	close(netChan)

	wg.Wait()

	return nil
}
