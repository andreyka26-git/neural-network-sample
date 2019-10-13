using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkLib
{
    public class MultipleOutputNet : NetBase
    {
        public MultipleOutputNet(List<NeuronLayer> layers)
            : base(layers)
        {
        }

        public int GetResult()
        {
            var outputLayer = GetOutputLayer();

            var answerIndex = 0;
            float max = 0;

            for(var neuronIndex = 0; neuronIndex < outputLayer.Neurons.Count; neuronIndex++)
            {
                if (outputLayer.Neurons[neuronIndex].Value > max)
                {
                    max = outputLayer.Neurons[neuronIndex].Value;
                    answerIndex = neuronIndex;
                }
            }

            //because indexing of array starts with 0, not with 1.
            return answerIndex - 1;
        }
    }
}
