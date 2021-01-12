using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace NeuralNetwork
{
    class Neural_Network
    {
        private int Isize { get; set; }
        private int Inodes { get; set; }
        private int Hnodes { get; set; }
        private int Onodes { get; set; }
        private int Hlayers { get; set; }
        private float Lr { get; set; }
        private float[,] Edata { get; set; }

        private float[] InputSignal { get; set; }
        private float[,] InputWeights { get; set; }
        private float[,,] HiddenWeights { get; set; }
        private float[,] OutputWeights { get; set; }
        private float[,] HnodeValues { get; set; }
        private float[] OnodeValues { get; set; }
        private float[,] HNodeErrors { get; set; }

        public Neural_Network(int inputSize, int hiddenNodes, int hiddenLayers, int outputNodes, float learningRate)
        {
            this.Isize = inputSize;
            this.Inodes = inputSize * inputSize;
            this.Hnodes = hiddenNodes;
            this.Hlayers = hiddenLayers;
            this.Onodes = outputNodes;
            this.Lr = learningRate;

            //this.InputSignal = new float[Inodes];
            this.InputWeights = RandomNumber.IRandomArray(Hnodes, Inodes);
            this.HiddenWeights = RandomNumber.RandomArray(Hnodes, Hnodes, Hlayers - 1);
            this.OutputWeights = RandomNumber.RandomArray(Onodes, Hnodes);
            this.HnodeValues = new float[Hlayers, Hnodes];
            this.OnodeValues = new float[Onodes];

            this.HNodeErrors = new float[hiddenLayers, hiddenNodes];
        }
         
        public Neural_Network(string preTrainedModelName)
        {
            PreTrainedModle(preTrainedModelName);
        }
         
        public void Train(string directory, float[,] targetOutput)
        {
            if ((targetOutput.GetLength(0) == Directory.GetFiles(directory).Length) && (targetOutput.GetLength(1) == Onodes))
            {
                this.Edata = targetOutput;
                try
                {
                    string[] items = Directory.GetFiles(directory);
                    int i = 0;
                    foreach (string item in items)
                    {
                        ForwardPropagate(item);
                        WeightRefine(i);
                        Array.Clear(HnodeValues, 0, HnodeValues.Length);
                        Array.Clear(OnodeValues, 0, OnodeValues.Length);
                        i++;
                    }

                }
                catch (DirectoryNotFoundException dirEx)
                {
                    throw dirEx;
                }
            }
            else
            {
                Console.WriteLine("expected output does not match, training don't work for a single image");
            }
        }   

        public float[] Run(string DetectImage)
        {
            Array.Clear(HnodeValues, 0, HnodeValues.Length);
            Array.Clear(OnodeValues, 0, OnodeValues.Length);

            return (ForwardPropagate(DetectImage));
        }

        
        // for running the NT.
        public float[] ForwardPropagate(string item)
        {
            InputSignal = Process.GraySignal(Process.ResizeImage(item, Isize)); 

            //first hidden node values.
            for (int j = 0; j < Hnodes; j++)
            {
                for (int i = 0; i < InputSignal.Length; i++)
                {
                    HnodeValues[0, j] += InputSignal[i] * InputWeights[i, j];
                }
                HnodeValues[0, j] = Operate.Sigmoid(HnodeValues[0, j]);
            }
             
            //setting hidden layers nodes values.
            for (int k = 0; k < Hlayers - 1; k++)
            {
                for (int j = 0; j < Hnodes; j++)
                {
                    for (int i = 0; i < Hnodes; i++)
                    {
                        HnodeValues[k + 1, j] += HnodeValues[k, i] * HiddenWeights[k, i, j];
                    }
                    HnodeValues[k + 1, j] = Operate.Sigmoid(HnodeValues[k + 1, j]);
                }
            }

            //setting output values.
            for (int j = 0; j < Onodes; j++)
            {
                for (int i = 0; i < Hnodes; i++)
                {
                    OnodeValues[j] += HnodeValues[Hlayers - 1, i] * OutputWeights[i, j];
                }
                OnodeValues[j] = Operate.Sigmoid(OnodeValues[j]);
            }           

            return OnodeValues;
        }

        // Updates weights.
        private void WeightRefine(int ImNum)
        {

            // filling the last layer of HNodesErrors.
            for (int j = 0; j < Hnodes; j++)
            {
                for (int i = 0; i < Onodes; i++)
                {
                    HNodeErrors[Hlayers - 1, j] += (Edata[ImNum, i] - OnodeValues[i]) * OutputWeights[j, i];
                }
                HNodeErrors[Hlayers - 1, j] /= 100;
            }

            // filling the HNodeErrors array.
            for (int k = 0; k <= Hlayers - 2; k++)
            {
                for (int j = 0; j < Hnodes; j++)
                {
                    for (int i = 0; i < Hnodes; i++)
                    {
                        HNodeErrors[Hlayers - 2 - k, j] += HNodeErrors[Hlayers - 1 - k, i] * HiddenWeights[Hlayers - 2 - k, j, i];
                    }
                    HNodeErrors[Hlayers - 2 - k, j] /= 100;
                }
            }

            //updating output weights.
            for (int j = 0; j < Hnodes; j++)
            {
                for (int i = 0; i < Onodes; i++)
                {
                    OutputWeights[j, i] += Lr * (Edata[ImNum, i] - OnodeValues[i]) * OnodeValues[i] * (1 - OnodeValues[i]) * HnodeValues[Hlayers - 1, j];
                }
            }

            //updating the Hidden Weights.  
            for (int k = 0; k <= Hlayers - 2; k++)
            {
                for (int j = 0; j < Hnodes; j++)
                {
                    for (int i = 0; i < Hnodes; i++)
                    {
                        HiddenWeights[Hlayers - 2 - k, j, i] += Lr * HNodeErrors[Hlayers - 1 - k, i] * HnodeValues[Hlayers - 1 - k, i] * (1 - HnodeValues[Hlayers - 1 - k, i]) * HnodeValues[Hlayers - 2 - k, j];
                    }
                }
            }

            //updating input weights.
            for (int j = 0; j < InputSignal.Length; j++)
            {
                for (int i = 0; i < Hnodes; i++)
                {
                    InputWeights[j, i] += Lr * HNodeErrors[0, i] * HnodeValues[0, i] * (1 - HnodeValues[0, i]) * InputSignal[j];
                }
            }

            Array.Clear(HNodeErrors, 0, HNodeErrors.Length);
        }

        // Saving a pretrained model
        public void SaveTrainedModel(string fileName)
        {
            if (!Directory.Exists(fileName))
            {
                using (StreamWriter file = new StreamWriter($"{fileName}_SData"))
                {
                    file.WriteLine(Isize);
                    file.WriteLine(Inodes);
                    file.WriteLine(Hnodes);
                    file.WriteLine(Onodes);
                    file.WriteLine(Hlayers);
                    file.WriteLine(Lr);
                }

                using (StreamWriter file = new StreamWriter($"{fileName}_IWeights"))
                {
                    for (int j = 0; j < InputWeights.GetLength(0); j++)
                    {
                        for (int i = 0; i < InputWeights.GetLength(1); i++)
                        {
                            file.Write($"{InputWeights[j, i]},");
                        }
                    }
                }

                using (StreamWriter file = new StreamWriter($"{fileName}_HWeights"))
                {
                    for (int k = 0; k < HiddenWeights.GetLength(0); k++)
                    {
                        for (int j = 0; j < HiddenWeights.GetLength(1); j++)
                        {
                            for (int i = 0; i < HiddenWeights.GetLength(2); i++)
                            {
                                file.Write($"{HiddenWeights[k, j, i]},");
                            }
                        }
                    }
                }

                using (StreamWriter file = new StreamWriter($"{fileName}_OWeights"))
                {
                    for (int j = 0; j < OutputWeights.GetLength(0); j++)
                    {
                        for (int i = 0; i < OutputWeights.GetLength(1); i++)
                        {
                            file.Write($"{OutputWeights[j, i]},");
                        }
                    }
                }
            }
            else
            {
                Console.WriteLine("current file name already available");
            }
        }

        // retriving a pretrained model
        private void PreTrainedModle(string fileName)
        {
            using(StreamReader read = new StreamReader($"{fileName}_SData"))
            {
                Isize = int.Parse(read.ReadLine());
                Inodes = int.Parse(read.ReadLine());
                Hnodes = int.Parse(read.ReadLine());
                Onodes = int.Parse(read.ReadLine());
                Hlayers = int.Parse(read.ReadLine());
                Lr = int.Parse(read.ReadLine());
            }
            InputSignal = new float[Inodes];
            InputWeights = new float[Inodes, Hnodes];
            HiddenWeights = new float[Hlayers - 1, Hnodes, Hnodes];
            OutputWeights = new float[Hnodes, Onodes];
            HnodeValues = new float[Hlayers, Hnodes];
            OnodeValues = new float[Onodes];

            using (StreamReader read = new StreamReader($"{fileName}_IWeights"))
            {
                string values = read.ReadToEnd();
                string[] Values = values.Split(',');
                int k = 0;
                for (int j = 0; j < Inodes; j++)
                {
                    for (int i = 0; i < Hnodes; i++)
                    {
                        InputWeights[j, i] = float.Parse(Values[k]);
                        k++;
                    }
                }
            }

            using (StreamReader read = new StreamReader($"{fileName}_HWeights"))
            {
                string values = read.ReadToEnd();
                string[] Values = values.Split(',');
                int l = 0;
                for (int k = 0; k < Hlayers - 1; k++)
                {
                    for (int j = 0; j < Hnodes; j++)
                    {
                        for (int i = 0; i < Hnodes; i++)
                        {
                            HiddenWeights[k, j, i] = float.Parse(Values[l]);
                            l++;
                        }
                    }
                }
            }

            using (StreamReader read = new StreamReader($"{fileName}_OWeights"))
            {
                string values = read.ReadToEnd();
                string[] Values = values.Split(',');
                int k = 0;
                for (int j = 0; j < Hnodes; j++)
                {
                    for (int i = 0; i < Onodes; i++)
                    {
                        OutputWeights[j, i] = float.Parse(Values[k]);
                        k++;
                    }
                }
            }

        }
    }
}
