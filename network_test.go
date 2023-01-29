package NeuralNetwork

import (
	"testing"
)

// used for presenting the network for the developer
/*func TestSchemaPrint(t *testing.T) {
	myNetwork, err := NewNeuralNetwork(1, []int{10, 20, 50, 60, 80, 100, 500, 100000, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	myNetwork.PrintNetworkSchema()
}*/

///////////////////////////////////////////////////////
////		   Creating network tests			   ////
///////////////////////////////////////////////////////

type testStructureData struct {
	nodesPerLayer []int
	outputLabels  []string
}

func TestGoodInputData(t *testing.T) {
	const numberOfNetworks = 1
	GoodStructureData := []testStructureData{
		{[]int{3, 10, 10, 10, 3}, []string{"a", "a", "a"}},
		{[]int{10, 20, 50, 60, 80, 100, 500, 1000, 1, 1}, []string{""}},
		{[]int{999, 1}, []string{""}},
	}
	for _, data := range GoodStructureData {
		myNetwork, err := NewNeuralNetwork(numberOfNetworks, data.nodesPerLayer, data.outputLabels)
		if err != nil {
			t.Fatal(data, err)
		}

		if len(myNetwork.networks[0].layers) != len(data.nodesPerLayer) {
			t.Fatal("Number of layers is wrong")
		}
		for i, layer := range myNetwork.networks[0].layers {
			if len(layer.nodes) != data.nodesPerLayer[i] {
				t.Fatal("Number of nodes per layer")
			}
		}
	}
}

func TestBadInputData(t *testing.T) {
	const numberOfNetworks = 1
	BadStructureData := []testStructureData{
		{[]int{}, []string{}},
		{[]int{10, 20, 0, 60, 0, 100, 500, 1000, 1}, []string{""}},
		{[]int{0}, []string{}},
		{[]int{-10}, []string{}},
		{[]int{3}, []string{""}},
		{[]int{3}, []string{"", "", "", "", ""}},
	}
	for _, data := range BadStructureData {
		_, err := NewNeuralNetwork(numberOfNetworks, data.nodesPerLayer, data.outputLabels)
		if err == nil {
			t.Fatal("Bad data got through: ", data.nodesPerLayer)
		}
	}
}

// tests wheter all networks are created with the same structure
func TestMultipleNetworks(t *testing.T) {
	const numberOfNetworks = 100
	data := testStructureData{[]int{10, 20, 50, 60, 80, 1, 1}, []string{""}}
	myNetwork, err := NewNeuralNetwork(numberOfNetworks, data.nodesPerLayer, data.outputLabels)
	if err != nil {
		t.Fatal(data, err)
	}

	for i := 1; i < numberOfNetworks; i++ {
		if len(myNetwork.networks[i].layers) != len(myNetwork.networks[0].layers) {
			t.Fatal("the networks don't have the same number of layers. Data: ", data)
		}
		for j := 0; j < len(myNetwork.networks[0].layers); j++ {
			if len(myNetwork.networks[i].layers[j].nodes) != len(myNetwork.networks[0].layers[j].nodes) {
				t.Fatal("the networks layers don't have the same number of nodes. Data: ", data)
			}
		}
	}
}

///////////////////////////////////////////////////////
////			Calculating output tests		   ////
///////////////////////////////////////////////////////

func TestCalculatingInputData(t *testing.T) {
	structureData := testStructureData{[]int{3, 20, 50, 60, 1, 1}, []string{""}}
	myNetwork, err := NewNeuralNetwork(10, structureData.nodesPerLayer, structureData.outputLabels)
	if err != nil {
		t.Fatal(structureData, err)
	}

	var badInputData = [][]float64{
		{0.5, 0.6, -1},
		{1.000001, 0.5, 0.6},
		{-0.000000000001, 0.5, 0.3},
		{0.5},
		{},
		{0.5, 0.6, 1, 0.5},
	}
	for _, data := range badInputData {
		_, err = myNetwork.GetOutputMap(data)
		if err == nil {
			t.Fatal("Bad calculating data got through: ", data)
		}
	}

	var goodInputData = [][]float64{
		{0.5, 0.6, 1},
		{1.000000, 0.5, 0.6},
		{0.000000000000, 0.5, 0.3},
		{0, 0.6, 1},
	}
	for _, data := range goodInputData {
		_, err = myNetwork.GetOutputMap(data)
		if err != nil {
			t.Fatal(data, err)
		}
	}
}
