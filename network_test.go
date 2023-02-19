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

		nodesPerLayer := myNetwork.network.GetNetworkStructure()
		if len(nodesPerLayer) != len(data.nodesPerLayer) {
			t.Fatal("Number of layers is wrong")
		}
		for i, nodes := range nodesPerLayer {
			if nodes != data.nodesPerLayer[i] {
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

func TestTheBestOutput(t *testing.T) {
	structureData := testStructureData{[]int{3, 20, 50, 60, 1, 5}, []string{"", "", "", "", ""}}
	myNetwork, err := NewNeuralNetwork(10, structureData.nodesPerLayer, structureData.outputLabels)
	if err != nil {
		t.Fatal(structureData, err)
	}
	var goodInputData = [][]float64{
		{0.5, 0.6, 1},
		{1.000000, 0.5, 0.6},
		{0.000000000000, 0.5, 0.3},
		{0, 0.6, 1},
	}
	for _, data := range goodInputData {
		theMap, _ := myNetwork.GetOutputMap(data)
		bestLabel, _ := myNetwork.GetNetworkResult(data)
		for _, v := range theMap {
			if v > theMap[bestLabel] {
				t.Fatal("the best label is not the best", data)
			}
		}
		newBestLabel, _ := myNetwork.GetNetworkResult(data)
		if newBestLabel != bestLabel {
			t.Fatal("calculating over the same data gives diffrent results", data)
		}
	}
}

/////////////////////////////////////////////////////////////
////			   User Data Load Tests					 ////
/////////////////////////////////////////////////////////////

func TestDataSetsLoad(t *testing.T) {
	structureData := testStructureData{[]int{3, 20, 50, 60, 1, 1}, []string{"welp"}}
	myNetwork, err := NewNeuralNetwork(10, structureData.nodesPerLayer, structureData.outputLabels)
	if err != nil {
		t.Fatal(structureData, err)
	}

	type data struct {
		input          [][]float64
		expectedOutput []string
	}

	goodData := []data{
		{[][]float64{{1, 0.5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"welp", "welp", "welp"}},
		{[][]float64{{1, 0.5, 0.6}}, []string{"welp"}},
		{[][]float64{{1, 0, 0.6}}, []string{"welp"}},
	}
	for _, myData := range goodData {
		err := myNetwork.LoadTrainingData(myData.input, myData.expectedOutput)
		if err != nil {
			t.Fatal(myData, err)
		}
	}

	badData := []data{
		{[][]float64{{1, 5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"welp", "welp", "welp"}},
		{[][]float64{{1, 1, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"welp", "kiss", "welp"}},
		{[][]float64{{1, 0.5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"welp", "welp"}},
		{[][]float64{{1, 0.5, 0.6}}, []string{"welp", "welp"}},
		{[][]float64{{1, -0.5, 0.6}}, []string{"welp"}},
		{[][]float64{{1, 0.5, 0.6, 1}}, []string{"welp"}},
		{[][]float64{{1, 0.5, 0.6, 1}}, []string{"wel"}},
	}

	for _, myData := range badData {
		err := myNetwork.LoadTrainingData(myData.input, myData.expectedOutput)
		if err == nil {
			t.Fatal("bad data got through: ", myData)
		}
	}
}

func TestDataSingleAddition(t *testing.T) {
	structureData := testStructureData{[]int{3, 20, 50, 60, 1, 1}, []string{"welp"}}
	myNetwork, err := NewNeuralNetwork(10, structureData.nodesPerLayer, structureData.outputLabels)
	if err != nil {
		t.Fatal(structureData, err)
	}

	type data struct {
		input          []float64
		expectedOutput string
	}

	goodData := []data{
		{[]float64{0.1, 0.3, 0.5}, "welp"},
		{[]float64{0.1, 0, 0.5}, "welp"},
		{[]float64{0.1, 1, 0.5}, "welp"},
	}
	for _, myData := range goodData {
		err := myNetwork.AddSingleTrainingData(myData.input, myData.expectedOutput)
		if err != nil {
			t.Fatal(err, myData)
		}
	}

	badData := []data{
		{[]float64{0.1, 0, -0.5}, ""},
		{[]float64{0.1, 5, 0.5}, "adsatrheragg"},
		{[]float64{0.1, 0.6, 0.5, 1}, "welp"},
		{[]float64{0.1, 0.6, 0.5}, "wlp"},
	}
	for _, myData := range badData {
		err := myNetwork.AddSingleTrainingData(myData.input, myData.expectedOutput)
		if err == nil {
			t.Fatal("bad data got through: ", myData)
		}
	}
}

/////////////////////////////////////////////////////////////
////			    Calculating Cost Tests			     ////
/////////////////////////////////////////////////////////////

func TestCalculatingCosts(t *testing.T) {
	structureData := testStructureData{[]int{3, 20, 50, 60, 1, 3}, []string{"1", "2", "welp"}}
	myNetwork, err := NewNeuralNetwork(10, structureData.nodesPerLayer, structureData.outputLabels)
	if err != nil {
		t.Fatal(structureData, err)
	}

	type data struct {
		input          [][]float64
		expectedOutput []string
	}

	goodData := []data{
		{[][]float64{{1, 0.5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"1", "2", "1"}},
		{[][]float64{{1, 0.5, 0.6}}, []string{"2"}},
		{[][]float64{{1, 0, 0.6}}, []string{"1"}},
	}
	for _, myData := range goodData {
		err := myNetwork.LoadTrainingData(myData.input, myData.expectedOutput)
		if err != nil {
			t.Fatal(myData, err)
		}

		myNetwork.network.CalculateCost(myNetwork.trainingDataSets)
		if myNetwork.network.GetCost() < 0 {
			t.Fatal("calculated cost is incorect: ", myNetwork.network.GetCost(), myData)
		}
	}
}
