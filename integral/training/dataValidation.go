package training

import (
	"errors"

	"github.com/Basileus1990/NeuralNetwork.git/integral/network"
)

func (trainer *Trainer) validateTrainingInputData(inputs [][]float64, outputs []string) error {
	if len(inputs) != len(outputs) {
		return errors.New("number of inputs slices is not the same as number of outputs")
	}
	for _, input := range inputs {
		_, numberOfInputNodes := trainer.networksAndCosts[0].network.NetworkStructure()
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
