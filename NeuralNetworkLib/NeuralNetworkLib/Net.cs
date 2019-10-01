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
                    neuron.Value = MathHelper.Activation(sum + biasSynapse.Weight);
                }
            }

            var lastLayer = _layers.Last();
            var result = lastLayer.Neurons.Single();
            return result.Value;
        }

        public void CorrectNet(float target, float error, float learningRate)
        {
            for (var layerIndex = _layers.Count - 1; layerIndex > 0; layerIndex--)
            {
                foreach (var neuron in _layers[layerIndex].Neurons)
                {
                    var weightSynapses = neuron.InputSynapses.Take(neuron.InputSynapses.Count - 2);

                    foreach (var inputSynapse in weightSynapses)
                    {
                        inputSynapse.Weight = inputSynapse.Weight + (learningRate * (target - error) * inputSynapse.StartNeuron.Value);
                    }

                    var biasSynapse = neuron.InputSynapses.Skip(neuron.InputSynapses.Count - 2).Single();
                    biasSynapse.Weight = biasSynapse.Weight + learningRate * (target - error);
                }
            }
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
