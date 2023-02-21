package training

import (
	"runtime"
	"sort"
	"sync"

	"github.com/Basileus1990/NeuralNetwork.git/integral/network"
)

// TODO:
// Configuration (In the future from the json file)
const evolutionTraining = true

// const backPropagationTraining = false

type Trainer struct {
	networks         []network.Network
	numberOfNetworks int
	trainDataSets    network.DataSets
	Initialized      bool
}

// Initializes the trainer and creates training networks
func NewTrainer(originalNet network.Network, numberOfNet int) Trainer {
	var trainer Trainer
	trainer.numberOfNetworks = numberOfNet
	trainer.networks = append(trainer.networks, originalNet)
	// creates new training networks and initializes them
	for len(trainer.networks) < trainer.numberOfNetworks {
		var newNet network.Network
		newNet.InitializeNetwork(originalNet.GetNetworkStructure(), originalNet.GetOutputLabels())
		trainer.networks = append(trainer.networks, newNet)
	}

	trainer.Initialized = true
	return trainer
}

// trains the network iterations times with training dataset
func (trainer *Trainer) Train(dataSets network.DataSets, iterations int) error {
	trainer.trainDataSets = dataSets

	for i := 0; i < iterations; i++ {
		if evolutionTraining {
			err := trainer.evolutionTraining()
			if err != nil {
				return err
			}
		}
		// TODO: add back propagation support
		// if backPropagationTraining {
		// 	//w
		// }
	}
	return nil
}

// calculate concurrently an average cost for every network for all training datasets
// and add them to trainer's costs map
func calculateAverageCosts(networks *[]network.Network, dataSets network.DataSets) {
	numberOfWorkers := runtime.NumCPU()
	netChan := make(chan *network.Network)
	var wg sync.WaitGroup
	wg.Add(len(*networks))
	lock := sync.Mutex{}

	for i := 0; i < numberOfWorkers; i++ {
		go func(wg *sync.WaitGroup, netChan chan *network.Network) {
			for net := range netChan {
				net.CalculateCost(&lock, dataSets)
				wg.Done()
			}
		}(&wg, netChan)
	}

	for i := range *networks {
		netChan <- &(*networks)[i]
	}
	close(netChan)
	wg.Wait()
}

// returns networks [0] <-- the best [n] <-- worse
func getSortedNetworks(networks *[]network.Network) []*network.Network {
	m := make(map[float64]*network.Network)
	sortValues := make([]float64, len(*networks))
	sortedNetworks := make([]*network.Network, len(*networks))
	for i := range *networks {
		m[(*networks)[i].GetCost()] = &(*networks)[i]
		sortValues[i] = (*networks)[i].GetCost()
	}

	sort.Float64s(sortValues)
	for i := range sortedNetworks {
		sortedNetworks[i] = m[sortValues[i]]
	}
	return sortedNetworks
}
