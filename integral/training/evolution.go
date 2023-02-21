package training

import (
	"errors"
	"math"
	"math/rand"

	"github.com/Basileus1990/NeuralNetwork.git/integral/network"
)

// a number determining the range of random mutations
const strengthOfEvolution = 10

// determines how much children are created in comparison to number of parents (0, inf)
// heavy performance impact
const percentageOfChildrenToParents = 0.5

// determines wheter the network will favour the best networks while mating
const favourBestNetworksWhileMating = true

// determines what weight will the network get -> every next gets multiplied by this number (0,1)
const maxNetworksSurvivorsWeight = 0.8

func (trainer *Trainer) evolutionTraining() error {
	calculateAverageCosts(&trainer.networks, trainer.trainDataSets)
	if favourBestNetworksWhileMating {
		err := trainer.createNewFavouredGeneration(getSortedNetworks(&trainer.networks))
		if err != nil {
			return err
		}
	} else {
		trainer.createNewRandomGeneration()
	}
	trainer.killWorstNetworks()

	return nil
}

func (trainer *Trainer) killWorstNetworks() {
	sortedNet := getSortedNetworks(&trainer.networks)
	newNetworks := make([]network.Network, trainer.numberOfNetworks)
	for i := range newNetworks {
		newNetworks[i] = *sortedNet[i]
	}
	trainer.networks = newNetworks
}

// Creates new child networks from the parents and appends them to the trainer
// Parents are chosen using weights -> the parent with the lowest cost has the advantage
func (trainer *Trainer) createNewFavouredGeneration(sortedNet []*network.Network) error {
	numberOfChildren := int(float64(len(trainer.networks)) * percentageOfChildrenToParents)
	children := make([]network.Network, 0, numberOfChildren)
	maxWeight := getMaxSurvivorWeight(len(trainer.networks))

	for i := 0; i < numberOfChildren; i++ {
		// loops until it finds two diffrent survivors
		first, err := getRandomWeightedIndex(len(trainer.networks), maxWeight)
		if err != nil {
			return err
		}
		second := -1
		for first != second {
			second, err = getRandomWeightedIndex(len(trainer.networks), maxWeight)
			if err != nil {
				return err
			}
		}
		children = append(children, createChildFromParents(*sortedNet[first], *sortedNet[second]))
	}
	calculateAverageCosts(&children, trainer.trainDataSets)
	trainer.networks = append(trainer.networks, children...)
	return nil
}

// Creates new child networks from the parents and appends them to the trainer
// Parents are chosen randomly
func (trainer *Trainer) createNewRandomGeneration() {
	numberOfChildren := int(float64(len(trainer.networks)) * percentageOfChildrenToParents)
	children := make([]network.Network, 0, numberOfChildren)
	for i := 0; i < numberOfChildren; i++ {
		// loops until it finds two diffrent survivors
		first := rand.Intn(len(trainer.networks))
		second := -1
		for first != second {
			second = rand.Intn(len(trainer.networks))
		}
		children = append(children, createChildFromParents(trainer.networks[first], trainer.networks[second]))
	}
	calculateAverageCosts(&children, trainer.trainDataSets)
	trainer.networks = append(trainer.networks, children...)
}

// returns an random(but using weights) number from [0,numberOfSurvivors)
func getRandomWeightedIndex(numberOfNet int, maxWeight float64) (int, error) {
	random := rand.Float64() * maxWeight
	currentWeight := 0.0
	for i := 0; i < numberOfNet; i++ {
		currentWeight += math.Pow(maxNetworksSurvivorsWeight, float64(i))
		if random <= currentWeight {
			return i, nil
		}
	}
	return 0, errors.New("random survivor -> run out of survivors")
}

// sets weights and biases of the given network with the values of parents
// the value taken from parents is taken from them  in ratio50/50
func createChildFromParents(first, second network.Network) network.Network {
	nodesPerLayer := first.GetNetworkStructure()
	var child network.Network
	child.InitializeEmptyNetwork(nodesPerLayer, first.GetOutputLabels())
	// iterating over layers
	for i := 0; i < len(nodesPerLayer); i++ {
		// iterating over all nodes in a layer
		for j := 0; j < nodesPerLayer[i]; j++ {
			setBias(&child, first, second, i, j)

			// iteratig over all node's weights
			for k := 0; i != len(nodesPerLayer)-1 && k < nodesPerLayer[i+1]; k++ {
				setWeight(&child, first, second, i, j, k)
			}
		}
	}
	return child
}

// sets the given network's bias with randomly(50/50) selected parent's bias
func setBias(net *network.Network, first, second network.Network, i, j int) {
	var newBias float64
	if rand.Intn(2) == 0 {
		newBias = first.GetNodeBias(i, j)
	} else {
		newBias = second.GetNodeBias(i, j)
	}
	randomMutation := (rand.Float64() - 0.5) * 2 * strengthOfEvolution
	newBias += randomMutation

	net.SetNodeBias(i, j, newBias)
}

// sets the given network's weight with randomly(50/50) selected parent's weight
func setWeight(net *network.Network, first, second network.Network, i, j, k int) {
	var newWeight float64
	if rand.Intn(2) == 0 {
		newWeight = first.GetNodeWeight(i, j, k)
	} else {
		newWeight = second.GetNodeWeight(i, j, k)
	}
	randomMutation := (rand.Float64() - 0.5) * 2 * strengthOfEvolution
	newWeight += randomMutation

	net.SetNodeWeight(i, j, k, newWeight)
}

// returns the maximal weight for a survivor for the amount of the survivors
func getMaxSurvivorWeight(amountOfSurvivors int) (maxSurvivorWeight float64) {
	for i := 0; i < amountOfSurvivors; i++ {
		maxSurvivorWeight += math.Pow(maxNetworksSurvivorsWeight, float64(i))
	}
	return maxSurvivorWeight
}
