using System.Collections.Generic;

namespace NeuralNetworkLib
{
    public class NeuronLayer
    {
        public List<Neuron> Neurons { get; set; }

        public void ConnectToInputLayer(NeuronLayer inputNeuronLayer)
        {
            foreach (var localNeuron in Neurons)
            {
                foreach (var inputNeuron in inputNeuronLayer.Neurons)
                {
                    localNeuron.ConnectToInputNeuron(inputNeuron, 0);
                }
            }
        }

        public void ConnectToOutputLayer(NeuronLayer outputNeuronLayer)
        {
            foreach (var localNeuron in Neurons)
            {
                foreach (var inputNeuron in outputNeuronLayer.Neurons)
                {
                    localNeuron.ConnectToInputNeuron(inputNeuron, 0);
                }
            }
        }
    }
}
