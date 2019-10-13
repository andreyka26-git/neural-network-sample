using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkLib
{
    /// <summary>
    /// Represents PerceptronNet which contains layer of neurons and synapses between them
    /// There is 3 layers: input, hidden and output with one neuron
    /// </summary>
    public class PerceptronNet
    {
        /// <summary>
        /// All layers which net contains
        /// </summary>
        private readonly List<NeuronLayer> _layers;

        /// <summary>
        /// Constructor which connect all layers between each other
        /// </summary>
        /// <param name="layers"></param>
        public PerceptronNet(List<NeuronLayer> layers)
        {
            for (var layerIndex = 0; layerIndex + 1 < layers.Count; layerIndex++)
            {
                layers[layerIndex].ConnectToOutputLayerWithBiases(layers[layerIndex + 1]);
            }

            _layers = layers;
        }

        /// <summary>
        /// Takes some inputs and calculate values on neurons
        /// </summary>
        /// <param name="inputs">Input values. Count should be equal with count of neurons of first layer (without bias).</param>
        /// <returns>Returns output layer if it is single output.</returns>
        public float FeedForward(float[] inputs)
        {
            var firstLayerNeurons = _layers.First().Neurons;

            //count - 1 because of bias
            if (firstLayerNeurons.Count - 1 != inputs.Length)
                throw new InvalidOperationException("Length of input is not correlate with input neurons in neural network.");

            //set inputs  to appropriate input neurons
            for (var neuronIndex = 0; neuronIndex < firstLayerNeurons.Count - 1; neuronIndex++)
                firstLayerNeurons[neuronIndex].Value = inputs[neuronIndex];

            //layer index initially 1, because we cannot feed forward the input layer.
            for (var layerIndex = 1; layerIndex < _layers.Count; layerIndex++)
            {
                foreach (var neuron in _layers[layerIndex].Neurons)
                {
                    if (neuron.InputSynapses.Count == 0)
                        continue;

                    var weightSynapses = neuron.InputSynapses.Take(neuron.InputSynapses.Count - 1);
                    var biasSynapseWeight = neuron.InputSynapses.Skip(neuron.InputSynapses.Count - 1).Take(1).Single().Weight;

                    //multiply all except last because it is bias.
                    var sum = weightSynapses.Sum(s => s.StartNeuron.Value * s.Weight);
                    neuron.Value = MathHelper.Sigmoid(sum + biasSynapseWeight);
                }
            }
            
            var responseNeuron = GetOutputNeuron();
            return responseNeuron.Value;
        }

        /// <summary>
        /// Calculates errors from values on neurons.
        /// </summary>
        /// <param name="target">Desired value</param>
        public void CalculateError(float target)
        {
            var responseNeuron = GetOutputNeuron();
            responseNeuron.Error = target - responseNeuron.Value;

            var hiddenLayer = GetHiddenLayer();

            foreach (var hiddenLayerNeuron in hiddenLayer.Neurons)
            {
                hiddenLayerNeuron.Error = 0;

                foreach (var synapse in hiddenLayerNeuron.OutputSynapses)
                    hiddenLayerNeuron.Error += synapse.Weight * responseNeuron.Error;
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
        /// <param name="target">Desired value</param>
        /// <param name="learningRate">Rate of adjusting of weight</param>
        public void BackPropagate(float target, float learningRate)
        {
            //wee need to go from right to left not vise versas
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
        
        /// <summary>
        /// Gets output neuron of perceptron, which contains response value
        /// </summary>
        /// <returns></returns>
        public Neuron GetOutputNeuron()
        {
            return _layers.Last().Neurons.Single();
        }

        /// <summary>
        /// Gets hidden layer
        /// </summary>
        /// <returns></returns>
        private NeuronLayer GetHiddenLayer()
        {
            return _layers.Skip(1).Take(1).Single();
        }
    }
}
