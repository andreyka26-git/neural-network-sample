namespace NeuralNetworkLib
{
    public class TrainingData<T>
    {
        public T Data { get; set; }

        //correct answer to that output
        public float Target { get; set; }
    }
}
