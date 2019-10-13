using System;
using System.Collections.Generic;
using System.Linq;

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

        public void ConnectToOutputLayerWithBiases(NeuronLayer outputNeuronLayer)
        {
            var random = new Random();

            var outputNeurons = outputNeuronLayer.Neurons.Count > 1
                ? outputNeuronLayer.Neurons.Take(outputNeuronLayer.Neurons.Count - 1)
                : outputNeuronLayer.Neurons;

            //leave the bias
            foreach (var outputNeuron in outputNeurons)
            {
                //need to connect bias to output layer too here.
                foreach (var inputNeuron in Neurons)
                {
                    inputNeuron.ConnectToOutputNeuron(outputNeuron, (float)random.NextDouble());
                }
            }
        }
    }
}
