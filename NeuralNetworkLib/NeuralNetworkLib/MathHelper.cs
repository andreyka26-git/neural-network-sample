using System;

namespace NeuralNetworkLib
{
    public static class MathHelper
    {
        public static float Activation(float input)
        {
            if (input > 0)
                return 1;

            return -1;
            //return 1.0f / (1.0f + (float)Math.Exp(-input));
        }
    }
}
