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
            return 1.0f / (1.0f + (float)Math.Exp(-input));
        }
    }
}
