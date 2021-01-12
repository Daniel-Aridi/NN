using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    class RandomNumber
    {
        // Generats a matrix of ramdom numbers R=]0, 1[ with a given number of rows and columns.
        public static float[,] RandomArray(int firstDimention, int secondDimention)
        {
            Random random = new Random();
            float[,] values = new float[secondDimention, firstDimention];
            for (int i = 0; i < secondDimention; i++)
            {
                for (int j = 0; j < firstDimention; j++)
                {
                    float num = random.Next(-9, 9);
                    if (num == 0)
                    {
                        float result = (num + 2 )/100;
                        values[i, j] = result; 
                    }
                    else
                    {
                        float result = num/100;
                        values[i, j] = result;
                    }
                }

            }
            return values;
        }

        public static float[,] IRandomArray(int firstDimention, int secondDimention)
        {
            Random random = new Random();
            float[,] values = new float[secondDimention, firstDimention];
            for (int i = 0; i < secondDimention; i++)
            {
                for (int j = 0; j < firstDimention; j++)
                {
                    float num = random.Next(-9, 9);
                    if (num == 0)
                    {
                        float result = (num + 2) / 100;
                        values[i, j] = result;
                    }
                    else
                    {
                        float result = num / 100;
                        values[i, j] = result;
                    }
                }

            }
            return values;
        }

        // Generats a matrix of ramdom numbers R=]0, 1[ of 3 dimentional array.
        public static float[,,] RandomArray(int firstDimention, int secondDimention, int thirdDimention)
        {
            Random random = new Random();
            float[,,] values = new float[thirdDimention, secondDimention, firstDimention];

            for (int k = 0; k < thirdDimention; k++)
            {
                for (int i = 0; i < secondDimention; i++)
                {
                    for (int j = 0; j < firstDimention; j++)
                    {
                        float num = random.Next(-9, 9);
                        if (num == 0)
                        {
                            float result = ((num + 2))/ 100;//(100- (k + 1));
                            values[k, i, j] = result; 
                        }
                        else
                        {
                            float result = (num) / 100;//(100 - (k + 1));
                            values[k, i, j] = result;
                        }
                    }

                }
            }
            return values;
        }
    }
}
