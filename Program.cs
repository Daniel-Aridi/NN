using System;
using System.Drawing;


namespace NeuralNetwork
{
    class Program
    {
        
        static void Main(string[] args)
        {

            //float[,] target = new float[,] { { 0.01f, 0.9f }, { 0.01f, 0.9f } };
            //float[,] target2 = new float[,] { { 0.9f, 0.01f }, { 0.9f, 0.01f } };
            //string dir = @"E:Test";
            //string dir2 = @"E:Test2";


            Neural_Network network = new Neural_Network(28, 80, 80, 2, 0.3f);



            //int i = 0;
            //while (i < 20)
            //{
            //    network.Train(dir, target);
            //    network.Train(dir2, target2);
            //    i++;
            //}

            var values = network.Run(@"E:Test\1.png");
            Console.WriteLine($"{ values[0]} {values[1]}");
            Console.WriteLine("");

            var values2 = network.Run(@"E:Test2\1.png");
            Console.WriteLine($"{ values2[0]} {values2[1]}");
            Console.WriteLine("");

            //var values4 = network.Run(@"E:Test\1.png");
            //Console.WriteLine($"{ values4[0]} {values4[1]}");
            //Console.WriteLine("");

            //var values3 = network.Run(@"E:test3\image1.jpg");
            //Console.WriteLine($"{ values3[0]} {values3[1]}");
            //Console.WriteLine("");



            //float[] one = new float[] { 1, 2, 3 };
            //float[] two = new float[] { 4, 5, 6 };
            //Console.WriteLine(one[0]);
            //one = two;
            //Console.WriteLine(one[0]);


            var one = network.ForwardPropagate(@"E:Test\1.png");
            foreach (var item in one)
            {
                Console.WriteLine(item);
            }

            var two = network.ForwardPropagate(@"E:Test2\1.png");
            foreach (var item in two)
            {
                Console.WriteLine(item);
            }
        }
    }
}
