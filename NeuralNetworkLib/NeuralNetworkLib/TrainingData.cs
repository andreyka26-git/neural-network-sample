namespace NeuralNetworkLib
{
    /// <summary>
    /// Training data structure with target value which is the true value, and Data which is an input
    /// </summary>
    /// <typeparam name="TD">Type of Data</typeparam>
    /// <typeparam name="TT">Type of Targets</typeparam>
    public class TrainingData<TD, TT>
    {
        /// <summary>
        /// Input data
        /// </summary>
        public TD Data { get; set; }

        /// <summary>
        /// The true value which should be with DATA input
        /// </summary>
        public TT Targets { get; set; }
    }
}
