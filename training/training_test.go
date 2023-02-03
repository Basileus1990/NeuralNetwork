package training

import (
	"testing"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

func TestDataSetsLoad(t *testing.T) {
	dummyNetwork := new(network.Network)
	dummyNetwork.InitializeNetwork([]int{3, 1, 1, 1})
	trainer, err := NewTrainer(&[]network.Network{*dummyNetwork}, []string{""})
	if err != nil {
		t.Fatal(err)
	}

	type data struct {
		input          [][]float64
		expectedOutput []string
	}

	goodData := []data{
		{[][]float64{{1, 0.5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"www", "www", "www"}},
		{[][]float64{{1, 0.5, 0.6}}, []string{""}},
		{[][]float64{{1, 0, 0.6}}, []string{""}},
	}
	for _, myData := range goodData {
		err := trainer.LoadTrainingData(myData.input, myData.expectedOutput)
		if err != nil {
			t.Fatal(myData, err)
		}
	}

	badData := []data{
		{[][]float64{{1, 5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"www", "www", "www"}},
		{[][]float64{{1, 0.5, 0.6}, {1, 0.5, 0.6}, {1, 0.5, 0.6}}, []string{"www", "www"}},
		{[][]float64{{1, 0.5, 0.6}}, []string{"", "ughiofdukhvnoiadh"}},
		{[][]float64{{1, -0.5, 0.6}}, []string{""}},
		{[][]float64{{1, 0.5, 0.6, 1}}, []string{""}},
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
	trainer, err := NewTrainer(&[]network.Network{*dummyNetwork}, []string{""})
	if err != nil {
		t.Fatal(err)
	}

	type data struct {
		input          []float64
		expectedOutput string
	}

	goodData := []data{
		{[]float64{0.1, 0.3, 0.5}, "sgda"},
		{[]float64{0.1, 0, 0.5}, ""},
		{[]float64{0.1, 1, 0.5}, "adsatrheragg"},
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
		{[]float64{0.1, 0.6, 0.5, 1}, "adsatrheragg"},
	}
	for _, myData := range badData {
		err := trainer.AddSingleTrainingData(myData.input, myData.expectedOutput)
		if err == nil {
			t.Fatal("bad data got through: ", myData)
		}
	}
}
