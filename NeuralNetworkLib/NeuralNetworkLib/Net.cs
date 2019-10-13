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

            //layer index initially 1, because we cannot feed forward the input layer.
            for (var layerIndex = 1; layerIndex < _layers.Count; layerIndex++)
            {
                var neurons = _layers[layerIndex].Neurons;
                
                for (var neuronIndex = 0; neuronIndex < neurons.Count; neuronIndex++)
                {
                    var neuron = neurons[neuronIndex];
                    var weightSynapses = neuron.InputSynapses.Take(neuron.InputSynapses.Count - 1);
                    
                    var biasSynapseWeight = (neuron.InputSynapses.Count == 0) ? 0 : neuron.InputSynapses.Skip(neuron.InputSynapses.Count - 1).Take(1).Single().Weight;

                    //multiply all except last because it is bias.
                    var sum = weightSynapses.Sum(s => s.StartNeuron.Value * s.Weight);
                    neuron.Value = MathHelper.Sigmoid(sum + biasSynapseWeight);
                }
            }

            var lastLayer = _layers.Last();
            var response = lastLayer.Neurons.Single();
            return response.Value;
        }

        public void CalculateError(float target)
        {
            var responseNeuron = _layers.Last().Neurons.Single();
            responseNeuron.Error = target - responseNeuron.Value;

            var hiddenLayer = _layers[_layers.Count - 2];

            foreach (var hiddenLayerNeuron in hiddenLayer.Neurons)
            {
                hiddenLayerNeuron.Error = 0;

                foreach (var synapse in hiddenLayerNeuron.OutputSynapses)
                    hiddenLayerNeuron.Error += synapse.Weight * responseNeuron.Error;
            }
        }

        public void BackPropagate(float target, float learningRate)
        {
            var reversedLayers = ((IEnumerable<NeuronLayer>) _layers).Reverse().ToList();

            //we have no need to calculate in the first layer of inputs, because it has no one input synapse
            foreach (var layer in reversedLayers.Take(reversedLayers.Count - 1))
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

        public Net(List<NeuronLayer> layers)
        {
            for (var layerIndex = 0; layerIndex + 1 < layers.Count; layerIndex++)
            {
                layers[layerIndex].ConnectToOutputLayerWithBiases(layers[layerIndex + 1]);
            }

            _layers = layers;
        }
    }
}
