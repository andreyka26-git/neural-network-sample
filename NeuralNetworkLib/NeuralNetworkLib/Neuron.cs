using System.Collections.Generic;

namespace NeuralNetworkLib
{
    public class Neuron
    {
        public float Value { get; set; }

        public List<Synapse> InputSynapses { get; set; } = new List<Synapse>();

        public List<Synapse> OutputSynapses { get; set; } = new List<Synapse>();

        public void ConnectToInputNeuron(Neuron inputNeuron, float weight)
        {
            var synapse = new Synapse(inputNeuron, this, weight);
            InputSynapses.Add(synapse);
            inputNeuron.OutputSynapses.Add(synapse);
        }

        public void ConnectToOutputNeuron(Neuron outputNeuron, float weight)
        {
            var synapse = new Synapse(this, outputNeuron, weight);
            OutputSynapses.Add(synapse);
            outputNeuron.InputSynapses.Add(synapse);
        }
    }
}
