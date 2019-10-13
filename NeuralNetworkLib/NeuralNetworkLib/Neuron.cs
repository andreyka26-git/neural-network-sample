using System.Collections.Generic;

namespace NeuralNetworkLib
{
    /// <summary>
    /// Represents neuron DS, which has value, error, input or output synapses
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// Value which is calculated when feed forwarding or input value.
        /// </summary>
        public float Value { get; set; }

        /// <summary>
        /// Value which is calculated when propagate an error
        /// </summary>
        public float Error { get; set; }

        /// <summary>
        /// List of input synapses (synapses from left layer)
        /// </summary>
        public List<Synapse> InputSynapses { get; set; } = new List<Synapse>();

        /// <summary>
        /// List of output synapses (synapses from right layer)
        /// </summary>
        public List<Synapse> OutputSynapses { get; set; } = new List<Synapse>();

        /// <summary>
        /// Connects to input (left) layer
        /// </summary>
        /// <param name="inputNeuron">Neuron form left(previous) layer</param>
        /// <param name="weight">Value of synapse between input and current neuron</param>
        public void ConnectToInputNeuron(Neuron inputNeuron, float weight)
        {
            var synapse = new Synapse(inputNeuron, this, weight);
            InputSynapses.Add(synapse);
            inputNeuron.OutputSynapses.Add(synapse);
        }

        /// <summary>
        /// Connects to output (right) layer
        /// </summary>
        /// <param name="outputNeuron">Neuron from right(next) layer</param>
        /// <param name="weight">Value of synapse between output and current neuron</param>
        public void ConnectToOutputNeuron(Neuron outputNeuron, float weight)
        {
            var synapse = new Synapse(this, outputNeuron, weight);
            OutputSynapses.Add(synapse);
            outputNeuron.InputSynapses.Add(synapse);
        }
    }
}
