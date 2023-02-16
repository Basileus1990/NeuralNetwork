package network

import "errors"

////////////////////////////////////////////////////////////////////////
//// Potentially can cause a performance drop - remove if neccesary ////
////////////////////////////////////////////////////////////////////////

func (net *Network) validateBiasIndex(layerIndex, nodeIndex int) error {
	layers, nodesPerLayer := net.NetworkStructure()
	if layerIndex < 0 || layerIndex >= layers {
		return errors.New("layer index out of range")
	}
	if nodeIndex < 0 || nodeIndex >= nodesPerLayer[layerIndex] {
		return errors.New("node index out of range")
	}

	return nil
}

func (net *Network) validateWeightIndex(layerIndex, nodeIndex, weightIndex int) error {
	_, nodesPerLayer := net.NetworkStructure()
	if err := net.validateBiasIndex(layerIndex, nodeIndex); err != nil {
		return err
	}
	// exludes last layer because it doesn't have any child nodes
	if layerIndex == len(nodesPerLayer)-1 {
		return errors.New("there are no weights in output layer")
	}
	if weightIndex < 0 || weightIndex >= nodesPerLayer[layerIndex+1] {
		return errors.New("weight index out of range")
	}

	return nil
}
