package training

import (
	"errors"
	"math"
	"math/rand"

	"github.com/Basileus1990/NeuralNetwork.git/network"
)

// a number determining the range of random mutations
const strengthOfEvolution = 0.1

// determines what weight will the network get -> every next gets multiplied by this number (0,1)
const maxNetworksSurvivorsWeight = 0.8

// a percentage of how much of a generation survives (0,1)
const selectionHarshness = 0.5

func (trainer *Trainer) evolutionTraining() error {
	trainer.calculateAverageCosts()
	survivors, err := trainer.selectNetworksToSurvive()
	if err != nil {
		return err
	}
	err = trainer.createNewGeneration(survivors)
	if err != nil {
		return err
	}

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
func (trainer *Trainer) createNewGeneration(survivors []network.Network) error {
	maxSurvivorWeight := getMaxSurvivorWeight(len(survivors))
	for i := range trainer.networksAndCosts {
		// loops until it finds two diffrent survivors
		firstNetIndex, err := getRandomSurvivorIndex(len(survivors), maxSurvivorWeight)
		secondNetIndex := -1
		if err != nil {
			return err
		}
		for {
			secondNetIndex, err = getRandomSurvivorIndex(len(survivors), maxSurvivorWeight)
			if err != nil {
				return err
			}
			if firstNetIndex != secondNetIndex {
				break
			}
		}
		createChildFromSurvivors(trainer.networksAndCosts[i].network, survivors[firstNetIndex], survivors[secondNetIndex])
	}
	return nil
}

// returns an random(but using weights) number from [0,numberOfSurvivors)
func getRandomSurvivorIndex(numberOfSurvivors int, maxSurvivorWeight float64) (randomSurvivorIndex int, err error) {
	random := rand.Float64() * maxSurvivorWeight
	currentWeight := 0.0
	for i := 0; i < numberOfSurvivors; i++ {
		currentWeight += math.Pow(maxNetworksSurvivorsWeight, float64(i))
		if random <= currentWeight {
			return i, nil
		}
	}
	return 0, errors.New("random survivor -> run out of survivors")
}

// sets weights and biases of the given network with the values of parents
// the value taken from parents is taken from them  in ratio50/50
func createChildFromSurvivors(net *network.Network, firstParent network.Network, secondParent network.Network) error {
	numberOfLayers, nodesPerLayer := net.NetworkStructure()
	// iterating over layers
	for i := 0; i < numberOfLayers; i++ {
		// iterating over all nodes in a layer
		for j := 0; j < nodesPerLayer[i]; j++ {
			setBias(net, firstParent, secondParent, i, j)

			// iteratig over all node's weights
			for k := 0; i != numberOfLayers-1 && k < nodesPerLayer[i+1]; k++ {
				setWeight(net, firstParent, secondParent, i, j, k)
			}
		}
	}
	return nil
}

// sets the given network's bias with randomly(50/50) selected parent's bias
func setBias(net *network.Network, firstParent network.Network, secondParent network.Network, i, j int) error {
	var newBias float64
	var err error
	if rand.Intn(2) == 0 {
		newBias, err = firstParent.GetNodeBias(i, j)
	} else {
		newBias, err = secondParent.GetNodeBias(i, j)
	}
	if err != nil {
		return err
	}
	randomMutation := (rand.Float64() - 0.5) * 2 * strengthOfEvolution
	newBias += randomMutation

	err = net.SetNodeBias(i, j, newBias)
	if err != nil {
		return err
	}

	return nil
}

// sets the given network's weight with randomly(50/50) selected parent's weight
func setWeight(net *network.Network, firstParent network.Network, secondParent network.Network, i, j, k int) error {
	var newWeight float64
	var err error
	if rand.Intn(2) == 0 {
		newWeight, err = firstParent.GetNodeWeight(i, j, k)
	} else {
		newWeight, err = secondParent.GetNodeWeight(i, j, k)
	}
	if err != nil {
		return err
	}
	randomMutation := (rand.Float64() - 0.5) * 2 * strengthOfEvolution
	newWeight += randomMutation

	err = net.SetNodeWeight(i, j, k, newWeight)
	if err != nil {
		return err
	}

	return nil
}

// returns the maximal weight for a survivor for the amount of the survivors
func getMaxSurvivorWeight(amountOfSurvivors int) (maxSurvivorWeight float64) {
	for i := 0; i < amountOfSurvivors; i++ {
		maxSurvivorWeight += math.Pow(maxNetworksSurvivorsWeight, float64(i))
	}
	return maxSurvivorWeight
}
