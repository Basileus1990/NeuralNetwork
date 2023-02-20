package NeuralNetwork

import "errors"

func (neuralNet *neuralNetwork) validateInputData(inputData []float64) error {
	if len(inputData) != neuralNet.NumberOfInputNodes() {
		return errors.New("number of input data has to be the same as number of input nodes")
	}
	for _, v := range inputData {
		if v < 0 || v > 1 {
			return errors.New("network input has to be beetween [0,1]")
		}
	}
	return nil
}

func (neuralNet *neuralNetwork) validateTrainingInputData(inputs [][]float64, outputs []string) error {
	if len(inputs) != len(outputs) {
		return errors.New("number of inputs slices is not the same as number of outputs")
	}
	for _, input := range inputs {
		if err := neuralNet.validateInputData(input); err != nil {
			return err
		}
	}
	// checks if user given expected outputs were exist in trainer.outputLabels
	for _, output := range outputs {
		exists := false
		for _, v := range neuralNet.network.GetOutputLabels() {
			if v == output {
				exists = true
				break
			}
		}
		if !exists {
			return errors.New("given output doesn't exist: " + output)
		}
	}

	return nil
}

func validateNetworkInit(numberOfTrainingNetworks int, nodesPerLayer []int, outputLabels []string) error {
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
