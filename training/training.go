package training

import (
	"errors"

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
}

type dataSet struct {
	input          []float64
	expectedOutput string // a desired output label
}

func NewTrainer(networks *[]network.Network, outputLabels []string) (*Trainer, error) {
	if err := validateInitializationData(networks, outputLabels); err != nil {
		return nil, err
	}

	trainer := new(Trainer)
	for _, v := range *networks {
		trainer.networks = append(trainer.networks, &v)
	}
	trainer.outputLabels = outputLabels

	return trainer, nil
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
