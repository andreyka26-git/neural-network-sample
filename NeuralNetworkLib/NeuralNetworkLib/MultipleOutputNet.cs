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

            var answerIndex = MathHelper.GetIndexOfMaxValue(outputLayer.Neurons.Select(n => n.Value).ToArray());

            //because indexing of array starts with 0, not with 1.
            return answerIndex + 1;
        }
    }
}
