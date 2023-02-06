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
	var survivors []int
	sortedNetworks := trainer.getSortedNetworks()

	deadOnes := make([]int, rand.Intn(len(trainer.networksAndCosts)))
	for i := range deadOnes {
		deadOnes[i] = rand.Intn(len(trainer.networksAndCosts))
	}

	for i := 0; i < len(sortedNetworks) && i < numberToSurvive; i++ {
		isDead := false
		for _, v := range deadOnes {
			if v == i {
				isDead = true
				break
			}
		}
		if isDead {
			continue
		}
		survivors = append(survivors, i)
	}

	return survivors
}
