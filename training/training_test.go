package training

import (
	"testing"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

/////////////////////////////////////////////////////////////
////			   User Data Load Tests					 ////
/////////////////////////////////////////////////////////////

func TestDataSetsLoad(t *testing.T) {
	dummyNetwork := new(network.Network)
	dummyNetwork.InitializeNetwork([]int{3, 1, 1, 1})
	trainer, err := NewTrainer(&[]network.Network{*dummyNetwork}, []string{"welp"})
	if err != nil {
		t.Fatal(err)
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
		err := trainer.LoadTrainingData(myData.input, myData.expectedOutput)
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
		err := trainer.LoadTrainingData(myData.input, myData.expectedOutput)
		if err == nil {
			t.Fatal("bad data got through: ", myData)
		}
	}
}

func TestDataSingleAddition(t *testing.T) {
	dummyNetwork := new(network.Network)
	dummyNetwork.InitializeNetwork([]int{3, 1, 1, 1})
	trainer, err := NewTrainer(&[]network.Network{*dummyNetwork}, []string{"welp"})
	if err != nil {
		t.Fatal(err)
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
		err := trainer.AddSingleTrainingData(myData.input, myData.expectedOutput)
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
		err := trainer.AddSingleTrainingData(myData.input, myData.expectedOutput)
		if err == nil {
			t.Fatal("bad data got through: ", myData)
		}
	}
}

/////////////////////////////////////////////////////////////
////			    Calculating Cost Tests			     ////
/////////////////////////////////////////////////////////////

func TestCalculatingCosts(t *testing.T) {
	dummyNetwork := make([]network.Network, 50)
	for i := range dummyNetwork {
		dummyNetwork[i].InitializeNetwork([]int{3, 1, 10, 50, 1, 3})
	}
	trainer, err := NewTrainer(&dummyNetwork, []string{"1", "2", "3"})
	if err != nil {
		t.Fatal(err)
	}

	type data struct {
		input          [][]float64
		expectedOutput []string
	}

	goodData := []data{
		{[][]float64{{1, 0.5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"1", "2", "1"}},
		{[][]float64{{1, 0.5, 0.6}}, []string{"2"}},
		{[][]float64{{1, 0, 0.6}}, []string{"3"}},
	}
	for _, myData := range goodData {
		err := trainer.LoadTrainingData(myData.input, myData.expectedOutput)
		if err != nil {
			t.Fatal(myData, err)
		}

		trainer.calculateAverageCosts()
		for _, v := range trainer.costs {
			if v <= 0 {
				t.Fatal("calculated cost is incorect: ", v)
			}
		}
	}
}
