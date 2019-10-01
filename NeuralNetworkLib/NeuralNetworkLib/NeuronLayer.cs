using System;
using System.Collections.Generic;

namespace NeuralNetworkLib
{
    public class NeuronLayer
    {
        public List<Neuron> Neurons { get; set;  }
        
        public void ConnectToInputLayer(NeuronLayer inputNeuronLayer)
        {
            var random = new Random();

            foreach (var localNeuron in Neurons)
            {
                foreach (var inputNeuron in inputNeuronLayer.Neurons)
                {
                    localNeuron.ConnectToInputNeuron(inputNeuron, (float)random.NextDouble());
                }
            }
        }

        public void ConnectToOutputLayer(NeuronLayer outputNeuronLayer)
        {
            var random = new Random();

            foreach (var localNeuron in Neurons)
            {
                foreach (var outputNeuron in outputNeuronLayer.Neurons)
                {
                    localNeuron.ConnectToOutputNeuron(outputNeuron, (float)random.NextDouble());
                }
            }
        }
    }
}
