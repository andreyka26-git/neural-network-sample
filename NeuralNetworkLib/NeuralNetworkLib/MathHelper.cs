using System;

namespace NeuralNetworkLib
{
    public static class MathHelper
    {
        /// <summary>
        /// Activation Function to put value in (0 to 1) range
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static float Sigmoid(float input)
        {
            var response = 1.0f / (1.0f + (float)Math.Exp(-input));
            
            return response;
        }

        public static int GetIndexOfMaxValue(float[] values)
        {
            var answerIndex = 0;
            float max = 0;

            for (var neuronIndex = 0; neuronIndex < values.Length; neuronIndex++)
            {
                if (values[neuronIndex] > max)
                {
                    max = values[neuronIndex];
                    answerIndex = neuronIndex;
                }
            }

            return answerIndex;
        }
    }
}
