using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworkLib;

namespace ConsoleNandGateSample
{
    class Program
    {
        private static float _learningRate = (float) 0.4;
        static void Main()
        {
            var net = BuildNet();
            for (var epochIndex = 0; epochIndex < 5000; epochIndex++)
            {
                var errors = new List<float>();

                var trainingDataSet = GetXorTrainingData();

                //suppose that we need to get response form one output
                foreach (var trainingData in trainingDataSet)
                {
                    var output = net.FeedForward(trainingData.Data);

                    net.CalculateError(new [] { trainingData.Targets });

                    errors.Add(trainingData.Targets - output.Single().Value);

                    net.BackPropagate(new [] { trainingData.Targets }, _learningRate);
                }

                var averageError = errors.Sum() / errors.Count;
                Console.WriteLine(averageError);
            }

            Console.WriteLine("Training is ended.");

            while (true)
            {
                Console.WriteLine("Enter first value for Xor: ");
                var firstValue = float.Parse(Console.ReadLine() ?? throw new InvalidOperationException());

                Console.WriteLine("Enter second value for Xor: ");
                var secondValue = float.Parse(Console.ReadLine() ?? throw new InvalidOperationException());

                if (firstValue > 1 || firstValue < 0 || secondValue > 1 || secondValue < 0)
                {
                    Console.WriteLine("Bad input, exit.");
                    break;
                }

                var inputs = new[] {firstValue, secondValue};

                var response = net.FeedForward(inputs);
                var absoluteResponse = response.Single().Value > 0.5 ? 1 : 0;
                Console.WriteLine($"Response: {absoluteResponse}{Environment.NewLine}");
            }
        }

        static List<TrainingData<float[], float>> GetXorTrainingData()
        {
            var trainingDataFilePath = $"{Environment.CurrentDirectory}/Resources/XOR_training_data.txt";

            var trainingDataSet = new List<TrainingData<float[], float>>();
            var lines = File.ReadAllLines(trainingDataFilePath);

            foreach (var line in lines)
            {
                var splitLine = line.Split(" ");

                var firstInput = float.Parse(splitLine[0]);
                var secondInput = float.Parse(splitLine[1]);
                var target = float.Parse(splitLine[2]);

                var trainingData = new TrainingData<float[], float>
                {
                    Data = new[] { firstInput, secondInput },
                    Targets = target
                };

                trainingDataSet.Add(trainingData);
            }

            return trainingDataSet;
        }

        static List<TrainingData<float[], float>> GetNandTrainingData()
        {
            var trainingDataFilePath = $"{Environment.CurrentDirectory}/Resources/NAND_training_data.txt";

            var trainingDataSet = new List<TrainingData<float[], float>>();
            var lines = File.ReadAllLines(trainingDataFilePath);

            foreach (var line in lines)
            {
                var splitLine = line.Split(" ");

                var firstInput = float.Parse(splitLine[0]);
                var secondInput = float.Parse(splitLine[1]);
                var target = float.Parse(splitLine[2]);

                var trainingData = new TrainingData<float[], float>
                {
                    Data = new[] { firstInput, secondInput },
                    Targets = target
                };

                trainingDataSet.Add(trainingData);
            }

            return trainingDataSet;
        }

        // Build network with 2 inputs and one output (simple Perceptron)
        //   I     H       O

        //   O----- O -
        //    -  -      -
        //      -         - O
        //    -  -      -
        //   O----- O  -
        //      -     -
        //   O -    O

        //   B      B
        static PerceptronNet BuildNet()
        {
            var layers = new List<NeuronLayer>();

            var inputLayer = new NeuronLayer
            {
                Neurons = new List<Neuron>
                {
                    new Neuron(),
                    new Neuron(),
                    new Neuron // bias
                    {
                        Value = 1
                    }
                }
            };

            var hiddenLayer = new NeuronLayer
            {
                Neurons = new List<Neuron>
                {
                    new Neuron(),
                    new Neuron(),
                    new Neuron // bias
                    {
                        Value = 1
                    }
                }
            };

            var outputLayer = new NeuronLayer
            {
                Neurons = new List<Neuron>
                {
                    new Neuron()
                }
            };

            layers.Add(inputLayer);
            layers.Add(hiddenLayer);
            layers.Add(outputLayer);

            var network = new PerceptronNet(layers);
            return network;
        }
    }
}
