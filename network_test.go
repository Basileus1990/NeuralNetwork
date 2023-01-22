package NeuralNetwork

import "testing"

func TestSchemaPrint(t *testing.T) {
	myNetwork, err := NewNeuralNetwork(1, 5, []int{3, 10, 10, 10, 3})
	if err != nil {
		t.Fatal(err)
	}
	myNetwork.PrintNetworkSchema()
}

func TestCreatingNetwork(t *testing.T) {
	const numberOfNetworks = 1
	type networkStructureData struct {
		numberOfLayers int
		nodesPerLayer  []int
	}
	GoodStructureData := []networkStructureData{
		{5, []int{3, 10, 10, 10, 3}},
		{10, []int{10, 20, 50, 60, 80, 100, 500, 1000, 1, 1}},
		{1, []int{999}},
	}
	BadStructureData := []networkStructureData{
		{0, []int{}},
		{10, []int{10, 20, 50, 60, 80, 100, 500, 1000, 1}},
		{1, []int{0}},
		{1, []int{-10}},
		{-10, []int{5, 5, 5, 5, 5, 5, 5, 5, 5, 5}},
	}

	for _, data := range GoodStructureData {
		myNetwork, err := NewNeuralNetwork(numberOfNetworks, data.numberOfLayers, data.nodesPerLayer)
		if err != nil {
			t.Fatal(data, err)
		}

		if len(myNetwork.networks[0].layers) != data.numberOfLayers {
			t.Fatal("Number of layers is wrong")
		}
		for i, layer := range myNetwork.networks[0].layers {
			if len(layer.nodes) != data.nodesPerLayer[i] {
				t.Fatal("Number of nodes per layer")
			}
		}
	}

	for _, data := range BadStructureData {
		_, err := NewNeuralNetwork(numberOfNetworks, data.numberOfLayers, data.nodesPerLayer)
		if err == nil {
			t.Fatal("Bad data got through: ", data)
		}
	}
}
