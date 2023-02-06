package training

import (
	"math/rand"
)

// a number determining the range of random mutations
const strengthOfEvolution = 1

// a percentage of how much of a generation survives
const selectionHarshness = 0.5

func (trainer *Trainer) evolutionTraining() {
	trainer.calculateAverageCosts()
	trainer.selectNetworksToSurvive()

}

// returns a slice of selected networks which will survive.
// it takes mostly the best ones but has some change for selecting worse ones instead of the best
func (trainer *Trainer) selectNetworksToSurvive() []int {
	numberToSurvive := int(float64(len(trainer.networksAndCosts)) * selectionHarshness)
	//survivors := make([]int, len(trainer.networks)-numberToSurvive)

	deadOnes := make([]int, rand.Intn(numberToSurvive))
	for i := range deadOnes {
		deadOnes[i] = rand.Intn(len(trainer.networksAndCosts))
	}

	return nil
}
