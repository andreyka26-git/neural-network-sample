namespace NeuralNetworkLib
{
    /// <summary>
    /// Training data structure with target value which is the true value, and Data which is an input
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class TrainingData<T>
    {
        /// <summary>
        /// Input data
        /// </summary>
        public T Data { get; set; }

        /// <summary>
        /// The true value which should be with DATA input
        /// </summary>
        public float Target { get; set; }
    }
}
