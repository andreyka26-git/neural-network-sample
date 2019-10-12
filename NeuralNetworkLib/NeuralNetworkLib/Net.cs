using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Security.Cryptography;

namespace NeuralNetworkLib
{
    public class Net
    {
        private readonly List<NeuronLayer> _layers;
        
        public float FeedForward(float[] inputs)
        {
            var firstLayer = _layers.First();

            //count - 1 because of bias
            if (firstLayer.Neurons.Count - 1 != inputs.Length)
                throw new InvalidOperationException("Length of input is not correlate with input neurons in neural network.");

            for (var neuronIndex = 0; neuronIndex < firstLayer.Neurons.Count - 1; neuronIndex++)
                firstLayer.Neurons[neuronIndex].Value = inputs[neuronIndex];

            for (var layerIndex = _layers.Count - 1; layerIndex >= 0; layerIndex--)
            {
                var neurons = _layers[layerIndex].Neurons;
                
                for (var neuronIndex = 0; neuronIndex < neurons.Count; neuronIndex++)
                {
                    var neuron = neurons[neuronIndex];
                    var weightSynapses = neuron.InputSynapses.Take(neuron.InputSynapses.Count - 2);
                    var biasSynapse = neuron.InputSynapses.Skip(neuron.InputSynapses.Count - 2).Single();

                    //multiply all except last because it is bias.
                    var sum = weightSynapses.Sum(s => s.StartNeuron.Value * s.Weight);
                    neuron.Value = MathHelper.Sigmoid(sum + biasSynapse.Weight);
                }
            }

            var lastLayer = _layers.Last();
            var result = lastLayer.Neurons.Single();
            return result.Value;
        }

        public void CalculateError(float target)
        {
            var responseNeuron = _layers.Last().Neurons.Single();
            responseNeuron.Error = target - responseNeuron.Value;

            var hiddenLayer = _layers[_layers.Count - 2];

            foreach (var hiddenLayerNeuron in hiddenLayer.Neurons)
            {
                float error = 0;

                foreach (var synapse in hiddenLayerNeuron.OutputSynapses)
                    error += synapse.Weight * responseNeuron.Error;
            }

        }

        public void BackPropagate(float target, float learningRate)
        {

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
