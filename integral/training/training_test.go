package training

import (
	"log"
	"math/rand"
	"testing"
	"time"

	"github.com/Basileus1990/NeuralNetwork.git/integral/network"
)

func createDummyNetworkTrainer() *Trainer {
	// for testing purposes
	rand.Seed(time.Now().UnixNano())

	var net network.Network
	net.InitializeNetwork([]int{3, 6, 3}, []string{"1", "2", "3"})
	trainer := NewTrainer(net, 20)
	return trainer
}

/////////////////////////////////////////////////////////////
////			    	Evolution Tests				     ////
/////////////////////////////////////////////////////////////

func TestCreatingNewGeneration(t *testing.T) {

	goodData := make([]network.DataSets, 3)
	data := network.Data{}
	data.SetData([]float64{1, 0.5, 0.6}, "1")
	goodData[0] = append(goodData[0], data)
	data.SetData([]float64{1, 0.5, 0.6}, "2")
	goodData[0] = append(goodData[0], data)
	data.SetData([]float64{1, 0.5, 0.6}, "3")
	goodData[0] = append(goodData[0], data)

	data.SetData([]float64{1, 1, 1}, "1")
	goodData[1] = append(goodData[1], data)

	data.SetData([]float64{0, 0, 0}, "3")
	goodData[2] = append(goodData[2], data)
	for _, myData := range goodData {
		trainer := createDummyNetworkTrainer()
		trainer.trainDataSets = goodData[0]

		trainer.calculateAverageCosts()
		err := trainer.createNewFavouredGeneration(getSortedNetworks(&trainer.networks))
		if err != nil {
			t.Fatal(err, myData)
		}
		if len(trainer.networks) != trainer.numberOfNetworks+int(float64(trainer.numberOfNetworks)*percentageOfChildrenToParents) {
			t.Fatal("number of new generation networks is incorrect: ", len(trainer.networks)-trainer.numberOfNetworks)
		}
	}
}

// generates random colors and returns the color and wheter it is red or not
func createTrainingData(numberOfData int) network.DataSets {
	var dataSets network.DataSets
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
		var data network.Data
		data.SetData(input, label)
		dataSets = append(dataSets, data)
	}

	return dataSets
}

func getNetworkAccuracy(trainer *Trainer) float64 {
	bestNet := getSortedNetworks(&trainer.networks)[0]
	for _, v := range getSortedNetworks(&trainer.networks) {
		log.Println(v.GetCost())
	}
	// for _, v := range trainer.networks {
	// 	log.Println(v.GetCost())
	// }
	log.Println("...")
	amountOfCorrect := 0
	for _, data := range trainer.trainDataSets {
		bestOutputLabel, _ := bestNet.GetBestOutput(data.GetInputs())
		if data.GetExpOutput() == bestOutputLabel {
			amountOfCorrect++
		}
	}
	return float64(amountOfCorrect) / float64(len(trainer.trainDataSets))
}

func TestEvolution(t *testing.T) {
	// for testing purposes
	rand.Seed(time.Now().UnixNano())

	var net network.Network
	net.InitializeNetwork([]int{3, 3, 2}, []string{"red", "notRed"})
	trainer := NewTrainer(net, 10)

	trainer.trainDataSets = createTrainingData(1000)

	trainer.calculateAverageCosts()
	beforeAccuracy := getNetworkAccuracy(trainer)
	for i := 0; i < 1000; i++ {
		trainer.evolutionTraining()
	}
	//trainer.trainDataSets = createTrainingData(1000)
	trainer.calculateAverageCosts()
	afterAccuracy := getNetworkAccuracy(trainer)

	log.Println("Before: ", beforeAccuracy, ", After: ", afterAccuracy)
	if afterAccuracy < beforeAccuracy {
		t.Fatal("Evolution - previous generations are better than new ones")
	}

}
