using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkLib
{
    /// <summary>
    /// Represents PerceptronNet which contains layer of neurons and synapses between them
    /// There is 3 layers: input, hidden and output with one neuron
    /// </summary>
    public class PerceptronNet : NetBase
    {
        /// <summary>
        /// Constructor which connect all layers between each other
        /// </summary>
        /// <param name="layers"></param>
        public PerceptronNet(List<NeuronLayer> layers)
            : base(layers)
        {
        }
        
        /// <summary>
        /// Gets output neuron of perceptron, which contains response value
        /// </summary>
        /// <returns></returns>
        public Neuron GetOutputNeuron()
        {
            return Layers.Last().Neurons.Single();
        }
    }
}
