using System;

namespace NeuralNetworkLib
{
    public static class MathHelper
    {
        public static float Activation(float input)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-input));
        }
    }
}
