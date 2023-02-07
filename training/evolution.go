package training

import (
	"math/rand"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

// a number determining the range of random mutations
// const strengthOfEvolution = 1.0

// a percentage of how much of a generation survives (0,1)
const selectionHarshness = 0.5

func (trainer *Trainer) evolutionTraining() error {
	trainer.calculateAverageCosts()
	survivors, err := trainer.selectNetworksToSurvive()
	if err != nil {
		return err
	}
	trainer.createNewGeneration(survivors)

	return nil
}

// returns a slice of copies of the networks which will survive
// it takes mostly the best ones but has some change for selecting worse ones instead of the best
func (trainer *Trainer) selectNetworksToSurvive() (survivors []network.Network, err error) {
	numberToSurvive := int(float64(len(trainer.networksAndCosts)) * selectionHarshness)
	sortedNetworks := trainer.getSortedNetworks()

	// randomly selects a indexes of networks to be killed
	deadOnes := make([]int, rand.Intn(len(trainer.networksAndCosts)-numberToSurvive))
	for i := range deadOnes {
		deadOnes[i] = rand.Intn(len(trainer.networksAndCosts))
	}

	for i := 0; i < len(sortedNetworks) && len(survivors) < numberToSurvive; i++ {
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
		netCopy, err := trainer.networksAndCosts[i].network.CopyNetwork()
		if err != nil {
			return nil, err
		}
		survivors = append(survivors, netCopy)
	}

	return survivors, nil
}

// replaces values from the original networks with mutated values from the survivors
// uses mating mechanic - for one new gen net takes values randomly from 2 survivors
// mates randomly but lowest cost networks have an advantage
func (trainer *Trainer) createNewGeneration(survivors []network.Network) {

}
