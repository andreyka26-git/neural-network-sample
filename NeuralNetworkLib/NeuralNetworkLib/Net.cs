using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkLib
{
    public class Net
    {
        private readonly List<NeuronLayer> _layers;
        
        public float FeedForward(float[] inputs)
        {
            var firstLayer = _layers.First();

            if (firstLayer.Neurons.Count != inputs.Length)
                throw new InvalidOperationException("Length of input is not correlate with input neurons in neural network.");

            for (var neuronIndex = 0; neuronIndex < firstLayer.Neurons.Count; neuronIndex++)
                firstLayer.Neurons[neuronIndex].Value = inputs[neuronIndex];

            for (var layerIndex = 0; layerIndex < _layers.Count - 1; layerIndex++)
            {
                foreach (var neuron in _layers[layerIndex + 1].Neurons)
                {
                    var sum = neuron.InputSynapses.Sum(s => s.StartNeuron.Value * s.Weight);
                    neuron.Value = MathHelper.Activation(sum);
                }
            }

            var lastLayer = _layers.Last();
            var result = lastLayer.Neurons.Single();
            return result.Value;
        }

        public Net(List<NeuronLayer> layers)
        {
            for (var layerIndex = 0; layerIndex + 1 < layers.Count; layerIndex++)
            {
                layers[layerIndex].ConnectToOutputLayer(layers[layerIndex + 1]);
            }

            _layers = layers;
        }
    }
}
