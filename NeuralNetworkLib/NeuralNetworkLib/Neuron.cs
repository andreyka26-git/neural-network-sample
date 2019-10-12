using System.Collections.Generic;

namespace NeuralNetworkLib
{
    public class Neuron
    {
        //Value which calculated during feed forwarding
        public float Value { get; set; }

        //Value which calculated during calculation of error in order to perform back propagation
        public float Error { get; set; }

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
