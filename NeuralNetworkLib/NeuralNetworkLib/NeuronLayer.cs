using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkLib
{
    /// <summary>
    /// Layer which contains neurons
    /// </summary>
    public class NeuronLayer
    {
        /// <summary>
        /// List of neurons which are inside layer
        /// The last neuron called bias, it is always has an 1 value
        /// </summary>
        public List<Neuron> Neurons { get; set;  }
        
        /// <summary>
        /// Connects current neuron layer with left (previous) layer
        /// It will connect all neurons of current layer and all neurons of left layer
        /// </summary>
        /// <param name="inputNeuronLayer">Layer from left side (previous)</param>
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

        /// <summary>
        /// Connects current neuron layer with right (next) layer
        /// It will connect all neurons of current layer and all neurons of right layer
        /// </summary>
        /// <param name="outputNeuronLayer">Layer from right side (next)</param>
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

        /// <summary>
        /// Connects current layer to output (right layer)
        /// It should not connect biases with previous layer
        /// </summary>
        /// <param name="outputNeuronLayer">Layer from the right side</param>
        /// <param name="max">Max value of weight</param>
        /// <param name="min">Min value of weight</param>
        public void ConnectToOutputLayerWithBiases(NeuronLayer outputNeuronLayer, double max, double min)
        {
            var random = new Random();

            IEnumerable<Neuron> outputNeurons;

            //handle case, when we have one neuron on the output
            if (outputNeuronLayer.Neurons.Count > 1)
                outputNeurons = outputNeuronLayer.Neurons.Take(outputNeuronLayer.Neurons.Count - 1);
            else
                outputNeurons = outputNeuronLayer.Neurons;
                
            //leave the bias
            foreach (var outputNeuron in outputNeurons)
            {
                //connect all neurons with bias from current layer to right layer
                foreach (var inputNeuron in Neurons)
                {
                    var value = random.NextDouble() * (max - min) + min;

                    inputNeuron.ConnectToOutputNeuron(
                        outputNeuron, (float)value);
                }
            }
        }
    }
}
