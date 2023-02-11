package training

import (
	"log"
	"math/rand"
	"testing"
	"time"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

func createDummyNetworkTrainer() (*Trainer, error) {
	// for testing purposes
	rand.Seed(time.Now().UnixNano())

	dummyNetwork := make([]network.Network, 50)
	for i := range dummyNetwork {
		dummyNetwork[i].InitializeNetwork([]int{3, 6, 3})
	}
	trainer, err := NewTrainer(&dummyNetwork, []string{"1", "2", "welp"})
	if err != nil {
		return trainer, err
	}
	return trainer, nil
}

/////////////////////////////////////////////////////////////
////			   User Data Load Tests					 ////
/////////////////////////////////////////////////////////////

func TestDataSetsLoad(t *testing.T) {
	trainer, err := createDummyNetworkTrainer()
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
	trainer, err := createDummyNetworkTrainer()
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
	trainer, err := createDummyNetworkTrainer()
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
		{[][]float64{{1, 0, 0.6}}, []string{"1"}},
	}
	for _, myData := range goodData {
		err := trainer.LoadTrainingData(myData.input, myData.expectedOutput)
		if err != nil {
			t.Fatal(myData, err)
		}

		trainer.calculateAverageCosts()
		for _, v := range trainer.networksAndCosts {
			if v.cost < 0 {
				t.Fatal("calculated cost is incorect: ", v.cost, myData)
			}
		}
	}
}

/////////////////////////////////////////////////////////////
////			    	Evolution Tests				     ////
/////////////////////////////////////////////////////////////

func TestSelectingOnesToSurvive(t *testing.T) {
	trainer, err := createDummyNetworkTrainer()
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
		{[][]float64{{1, 0, 0.6}}, []string{"1"}},
	}
	for _, myData := range goodData {
		err := trainer.LoadTrainingData(myData.input, myData.expectedOutput)
		if err != nil {
			t.Fatal(myData, err)
		}

		trainer.calculateAverageCosts()
		net, err := trainer.selectNetworksToSurvive()
		if err != nil {
			t.Fatal(err, myData)
		}
		if len(net) != int(float64(len(trainer.networksAndCosts))*selectionHarshness) {
			t.Fatal("number of survivors is incorrect: ", len(net))
		}
	}
}

// generates random colors and returns the color and wheter it is red or not
func createTrainingData(numberOfData int) (inputs [][]float64, expOutput []string) {
	for i := 0; i < numberOfData; i++ {
		input := make([]float64, 3)
		red := rand.Intn(256)
		green := rand.Intn(256)
		blue := rand.Intn(256)
		input[0] = float64(red) / 255
		input[1] = float64(green) / 255
		input[2] = float64(blue) / 255
		label := ""
		if red > (blue+green)*2 {
			label = "red"
		} else {
			label = "notRed"
		}
		inputs = append(inputs, input)
		expOutput = append(expOutput, label)
	}

	return inputs, expOutput
}

func getNetworkAccuracy(trainer *Trainer) float64 {
	previousBestCost := trainer.networksAndCosts[0].cost
	bestIndex := 0
	for i, v := range trainer.networksAndCosts {
		if previousBestCost > v.cost {
			previousBestCost = v.cost
			bestIndex = i
		}
	}
	amountOfCorrect := 0
	for _, data := range trainer.dataSets {
		trainer.networksAndCosts[bestIndex].network.CalculateOutput(nil, data.input)
		output := trainer.networksAndCosts[bestIndex].network.GetOutputValuesSlice()
		bestOutputIndex := 0
		for i := 1; i < len(output); i++ {
			if output[bestOutputIndex] < output[i] {
				bestOutputIndex = i
			}
		}
		if data.expectedOutput == trainer.outputLabels[bestOutputIndex] {
			amountOfCorrect++
		}
	}
	return float64(amountOfCorrect) / float64(len(trainer.dataSets))
}

func TestEvolution(t *testing.T) {
	// for testing purposes
	rand.Seed(time.Now().UnixNano())

	dummyNetwork := make([]network.Network, 50)
	for i := range dummyNetwork {
		dummyNetwork[i].InitializeNetwork([]int{3, 2})
	}
	trainer, err := NewTrainer(&dummyNetwork, []string{"red", "notRed"})
	if err != nil {
		t.Fatal(err)
	}

	inputs, expOutputs := createTrainingData(1000)
	err = trainer.LoadTrainingData(inputs, expOutputs)
	if err != nil {
		t.Fatal(err)
	}

	trainer.calculateAverageCosts()
	log.Println(getNetworkAccuracy(trainer))
	previousBestCost := trainer.networksAndCosts[0].cost
	for _, v := range trainer.networksAndCosts {
		if previousBestCost > v.cost {
			previousBestCost = v.cost
		}
	}
	for i := 0; i < 100; i++ {
		trainer.evolutionTraining()
		if err != nil {
			t.Fatal(err)
		}
	}
	trainer.calculateAverageCosts()
	newBestCost := trainer.networksAndCosts[0].cost
	for _, v := range trainer.networksAndCosts {
		if newBestCost > v.cost {
			newBestCost = v.cost
		}
	}

	log.Println(getNetworkAccuracy(trainer))
	if newBestCost > previousBestCost {
		t.Fatal("Evolution - previous generations are better than new ones")
	}

}
