namespace NeuralNetworkLib
{
    public class Synapse
    {
        public float Weight { get; set; }

        public Synapse(Neuron startNeuron, Neuron endNeuron, float weight)
        {
            StartNeuron = startNeuron;
            EndNeuron = endNeuron;
            Weight = weight;
        }

        public Neuron StartNeuron { get; }

        public Neuron EndNeuron { get; }
    }
}
