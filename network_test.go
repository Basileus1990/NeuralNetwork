package NeuralNetwork

import "testing"

func TestSchemaPrint(t *testing.T) {
	myNetwork, err := NewNeuralNetwork(1, 5, []int{3, 10, 10, 10, 3})
	if err != nil {
		t.Fatal(err)
	}
	myNetwork.PrintNetworkSchema()
}
