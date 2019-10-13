namespace NeuralNetworkLib
{
    /// <summary>
    /// Synapse which connects two neurons
    /// </summary>
    public class Synapse
    {
        /// <summary>
        /// Weight, represents strength of connection between StartNeuron and EndNeuron <see cref="Neuron"/>
        /// </summary>
        public float Weight { get; set; }

        /// <summary>
        /// Constructor for immutability
        /// </summary>
        /// <param name="startNeuron">Neuron from left layer</param>
        /// <param name="endNeuron">Neuron from right layer</param>
        /// <param name="weight">Strength of connection between two layers.</param>
        public Synapse(Neuron startNeuron, Neuron endNeuron, float weight)
        {
            StartNeuron = startNeuron;
            EndNeuron = endNeuron;
            Weight = weight;
        }

        /// <summary>
        /// Neuron from left layer
        /// </summary>
        public Neuron StartNeuron { get; }

        /// <summary>
        /// Neuron from right layer
        /// </summary>
        public Neuron EndNeuron { get; }
    }
}
