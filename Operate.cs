using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    class Operate
    {
        //Operates the sigmoid function for a given input.
        public static float Sigmoid(float input)
        {
            float result = (float)(1 / (1 + Math.Exp(-input)));

            return result;
        }
    }
}
