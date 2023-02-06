package training

import (
	"errors"
	"math"
	"runtime"
	"sync"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

// TODO:
// Configuration (In the future from the json file)
const evolutionTraining = true
const backPropagationTraining = false

type Trainer struct {
	networks     []*network.Network
	costs        map[*network.Network]float64
	outputLabels []string
	dataSets     []dataSet
	locker       sync.Mutex
}

type dataSet struct {
	input          []float64
	expectedOutput string // a desired output label
}

// creates, configures and returns a pointer for a Trainer
func NewTrainer(networks *[]network.Network, outputLabels []string) (*Trainer, error) {
	if err := validateInitializationData(networks, outputLabels); err != nil {
		return nil, err
	}

	trainer := new(Trainer)
	for _, v := range *networks {
		trainer.networks = append(trainer.networks, &v)
	}
	trainer.outputLabels = outputLabels
	trainer.costs = make(map[*network.Network]float64)

	return trainer, nil
}

// trains the network iterations times with training dataset
func (trainer *Trainer) Train(iterations int) error {
	if iterations <= 0 {
		return errors.New("number of iterations has to be bigger than ones")
	}
	if len(trainer.dataSets) == 0 {
		return errors.New("the training data hasn't been yet loaded")
	}

	for i := 0; i < iterations; i++ {
		if evolutionTraining {
			trainer.evolutionTraining()
		}
		// TODO: add back propagation support
	}
	return nil
}

// assings the given data to the trainer replacing old data
func (trainer *Trainer) LoadTrainingData(inputs [][]float64, outputs []string) error {
	if err := trainer.validateTrainingInputData(inputs, outputs); err != nil {
		return err
	}

	dataSets := make([]dataSet, len(inputs))
	for i := 0; i < len(dataSets); i++ {
		dataSets[i].input = inputs[i]
		dataSets[i].expectedOutput = outputs[i]
	}

	trainer.dataSets = dataSets
	return nil
}

func (trainer *Trainer) AddSingleTrainingData(input []float64, output string) error {
	if err := trainer.validateTrainingInputData([][]float64{input}, []string{output}); err != nil {
		return err
	}

	trainer.dataSets = append(trainer.dataSets, dataSet{input: input, expectedOutput: output})
	return nil
}

// a thread safe way to get data sets
// returns a pointer to a copy of a single data set input slice and an index of expectedOutput
func (trainer *Trainer) getDataSet(index int) (*[]float64, int, error) {
	trainer.locker.Lock()
	defer trainer.locker.Unlock()

	if index >= len(trainer.dataSets) {
		return nil, -1, errors.New("no more data sets")
	}

	var expectedOutputIndex int
	for i, v := range trainer.outputLabels {
		if v == trainer.dataSets[index].expectedOutput {
			expectedOutputIndex = i
			break
		}
	}

	inputCopy := make([]float64, len(trainer.dataSets[index].input))
	copy(inputCopy, trainer.dataSets[index].input)

	return &inputCopy, expectedOutputIndex, nil
}

// calculate concurrently an average cost for every network for all training datasets
// and add them to trainer's costs map
func (trainer *Trainer) calculateAverageCosts() {
	numberOfWorkers := runtime.NumCPU()
	netChan := make(chan *network.Network)
	var wg sync.WaitGroup
	wg.Add(len(trainer.networks))
	for i := 0; i < numberOfWorkers; i++ {
		go func(wg *sync.WaitGroup, netChan chan *network.Network) {
			for net := range netChan {
				trainer.averageCost(wg, net)
			}
		}(&wg, netChan)
	}
	for _, net := range trainer.networks {
		netChan <- net
	}
	close(netChan)

	wg.Wait()

}

// calculates average cost of a single network and ads it to trainer.costs map
func (trainer *Trainer) averageCost(wg *sync.WaitGroup, net *network.Network) {
	defer wg.Done()

	inputData, expectedOutIndex, err := trainer.getDataSet(0)
	i := 1
	cost := 0.0
	for ; err == nil; i++ {
		net.CalculateOutput(nil, *inputData)
		outputs := net.GetOutputValuesSlice()
		for j, v := range outputs {
			if j == expectedOutIndex {
				cost -= math.Pow(v, 2)
			} else {
				cost += math.Pow(v, 2)
			}
		}

		inputData, expectedOutIndex, err = trainer.getDataSet(i)
	}

	trainer.locker.Lock()
	trainer.costs[net] = cost / float64(i)
	trainer.locker.Unlock()
}

func (trainer *Trainer) validateTrainingInputData(inputs [][]float64, outputs []string) error {
	if len(inputs) != len(outputs) {
		return errors.New("number of inputs slices is not the same as number of outputs")
	}
	for _, input := range inputs {
		_, numberOfInputNodes := trainer.networks[0].NetworkStructure()
		if len(input) != numberOfInputNodes[0] {
			return errors.New("number of input values is not the same as number of input nodes")
		}

		for _, v := range input {
			if v < 0 || v > 1 {
				return errors.New("input values have to be between [0,1]")
			}
		}
	}
	// checks if user given expected outputs were exist in trainer.outputLabels
	for _, inputOutput := range outputs {
		exists := false
		for _, v := range trainer.outputLabels {
			if v == inputOutput {
				exists = true
				break
			}
		}
		if !exists {
			return errors.New("given output doesn't exist: " + inputOutput)
		}
	}

	return nil
}

func validateInitializationData(networks *[]network.Network, outputLabels []string) error {
	if networks == nil {
		return errors.New("passed wrong networks pointer: nil")
	}

	numberOfLayers, numberOfInputNodes := (*networks)[0].NetworkStructure()
	if numberOfInputNodes[numberOfLayers-1] != len(outputLabels) {
		return errors.New("number of networks is not the same as number of outputs")
	}
	return nil
}
