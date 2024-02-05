

this is just a basic overview for a successful neural network computation from scratch to easly detect hand written numbers; low but sucffient amout of hidden layers allows for fast training and ifficient computation

it is not fully organized by any means, just to prove a point.


# Neural-Network
c# neural network



Neural_Network network = new Neural_Network(28, 80, 80, 2, 0.3f); // initializing the network.

float[,] target = new float[,] { { 0.01f, 0.9f }, { 0.01f, 0.9f } }; // creating target output data for each image
float[,] target2 = new float[,] { { 0.9f, 0.01f }, { 0.9f, 0.01f } };
string dir = @"E:Test";
string dir2 = @"E:Test2";

// training the network.
int i = 0;
while (i < 20)
{
    network.Train(dir, target);
    network.Train(dir2, target2);
    i++;
}

// runing network and outputing values.
var values = network.Run(@"E:Test\1.png");
Console.WriteLine($"{ values[0]} {values[1]}");
Console.WriteLine("");

var values2 = network.Run(@"E:Test2\1.png");
Console.WriteLine($"{ values2[0]} {values2[1]}");
Console.WriteLine("");
