package network

type DataSet struct {
	inputs         []float64
	expectedOutput string
}

func (dataSet *DataSet) SetDataSet(inputs []float64, expectedOutput string) {
	dataSet.inputs = inputs
	dataSet.expectedOutput = expectedOutput
}

func (dataSet *DataSet) GetExpOutput() string {
	return dataSet.expectedOutput
}

func (dataSet *DataSet) GetInputs() []float64 {
	return dataSet.inputs
}
