using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetworkLib
{
    public class NetBase
    {
        /// <summary>
        /// All layers which net contains
        /// </summary>
        protected readonly List<NeuronLayer> Layers;

        /// <summary>
        /// Constructor which connect all layers between each other
        /// </summary>
        /// <param name="layers">Layers</param>
        public NetBase(List<NeuronLayer> layers)
        {
            for (var layerIndex = 0;  layerIndex + 1 < layers.Count;  layerIndex++)
            {
                var max = 1.0f / layers[layerIndex].Neurons.Count;
                layers[layerIndex].ConnectToOutputLayerWithBiases(layers[layerIndex + 1], 0.0001, max);
            }

            Layers = layers;
        }

        /// <summary>
        /// Takes some inputs and calculate values on neurons
        /// </summary>
        /// <param name="inputs">Input values. Count should be equal with count of neurons of first layer (without bias).</param>
        /// <returns>Returns output layer if it is single output.</returns>
        public List<Neuron> FeedForward(float[] inputs)
        {
            var firstLayerNeurons = Layers.First().Neurons;

            //count - 1 because of bias
            if (firstLayerNeurons.Count - 1 != inputs.Length)
                throw new InvalidOperationException("Length of input is not correlate with input neurons in neural network.");

            //set inputs  to appropriate input neurons
            for (var neuronIndex = 0; neuronIndex < firstLayerNeurons.Count - 1; neuronIndex++)
                firstLayerNeurons[neuronIndex].Value = inputs[neuronIndex];

            //layer index initially 1, because we cannot feed forward the input layer.
            for (var layerIndex = 1; layerIndex < Layers.Count; layerIndex++)
            {
                foreach (var neuron in Layers[layerIndex].Neurons)
                {
                    if (neuron.InputSynapses.Count == 0)
                        continue;

                    var weightSynapses = neuron.GetInputWeightSynapses();
                    var biasSynapseWeight = neuron.GetInputBiasSynapse().Weight;

                    //multiply all except last because it is bias.
                    var sum = weightSynapses.Sum(s => s.StartNeuron.Value * s.Weight);
                    neuron.Value = MathHelper.Sigmoid(sum + biasSynapseWeight);
                }
            }

            var responseLayer = GetOutputLayer();
            return responseLayer.Neurons;
        }

        /// <summary>
        /// Calculates errors from values on neurons.
        /// </summary>
        public void CalculateError(float[] targets)
        {
            //moving form right side to the left side
            var layers = GetReversedLayers(Layers);

            //1. determine outputLayer error
            var outputLayer = layers.First();
            
            var neurons = outputLayer.Neurons;
            
            if(targets.Length != neurons.Count - 1)
                throw  new InvalidDataException("Targets length and count of output neurons is not equal.");
            
            for (var neuronIndex = 0; neuronIndex < neurons.Count - 1; neuronIndex++)
            {
                var neuron = neurons[neuronIndex];
                var target = targets[neuronIndex];

                neuron.Error = target - neuron.Value;
            }

            //2. propagate error to hidden layers

            //get all layers without first and last
            foreach (var layer in layers.Skip(1).Take(layers.Count - 2))
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Error = 0;

                    foreach (var synapse in neuron.OutputSynapses)
                        neuron.Error += synapse.Weight * synapse.EndNeuron.Error;
                }
            }
        }

        /// <summary>
        /// Adjust all weights via back propagation algorithm
        /// deltaWho = L * Eo * Oo * (1 - Oo) * Oh
        /// Who - delta weight
        /// L - learning weight
        /// Eo - output error (from next layer)
        /// Oo - output value (from next layer)
        /// Oh - current output value (from current layer) 
        /// </summary>
        /// <param name="learningRate">Rate of adjusting of weight</param>
        public void BackPropagate(float learningRate)
        {
            //we need to go from right to left not vise versa
            var layers = GetReversedLayers(Layers);

            //we have NO need to calculate in the first layer of inputs, because it has no one input synapse
            foreach (var layer in layers.Take(layers.Count - 1))
            {
                foreach (var outputNeuron in layer.Neurons)
                {
                    foreach (var inputSynapse in outputNeuron.InputSynapses)
                    {
                        var deltaWeight = learningRate * outputNeuron.Error * outputNeuron.Value * (1 - outputNeuron.Value) * inputSynapse.StartNeuron.Value;
                        inputSynapse.Weight += deltaWeight;
                    }
                }
            }
        }

        /// <summary>
        /// Gets output layer
        /// </summary>
        /// <returns>NeuronLayer</returns>
        public NeuronLayer GetOutputLayer()
        {
            var outputLayer = Layers.Last();
            return outputLayer;
        }

        /// <summary>
        /// Get reversed list of layers
        /// Method is immutable
        /// </summary>
        /// <param name="layers">Input Layers. They will not be changed.</param>
        /// <returns>Reversed layers</returns>
        public List<NeuronLayer> GetReversedLayers(List<NeuronLayer> layers)
        {
            var reversed = ((IEnumerable<NeuronLayer>) layers).Reverse();
            return reversed.ToList();
        }
    }
}
