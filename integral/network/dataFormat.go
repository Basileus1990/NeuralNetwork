package network

import (
	"sync"
)

type Data struct {
	inputs         []float64
	expectedOutput string
}

func (dataSet *Data) SetData(inputs []float64, expectedOutput string) {
	dataSet.inputs = inputs
	dataSet.expectedOutput = expectedOutput
}

func (dataSet *Data) GetExpOutput() string {
	return dataSet.expectedOutput
}

func (dataSet *Data) GetInputs() []float64 {
	return dataSet.inputs
}

type DataSets []Data

// A thread safe way to get single data
// Returns a copy of a single data
func (dataSets DataSets) GetSafeDataSetCopy(lock *sync.Mutex, index int) Data {
	lock.Lock()
	defer lock.Unlock()

	expectedOutput := dataSets[index].GetExpOutput()
	inputs := dataSets[index].GetInputs()
	inputCopy := make([]float64, len(inputs))
	copy(inputCopy, inputs)

	var dataCopy Data
	dataCopy.SetData(inputCopy, expectedOutput)
	return dataCopy
}
