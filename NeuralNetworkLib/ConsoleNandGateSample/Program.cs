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

                var trainingDataSet = GetXORTrainingData();

                //suppose that we need to get response form one output
                foreach (var trainingData in trainingDataSet)
                {
                    var output = net.FeedForward(trainingData.Data);

                    net.CalculateError(trainingData.Target);

                    errors.Add(trainingData.Target - output);

                    net.BackPropagate(trainingData.Target, _learningRate);
                }

                var averageError = errors.Sum() / errors.Count;
                Console.WriteLine(averageError);
            }

            Console.WriteLine("Training is ended.");
        }

        static List<TrainingData<float[]>> GetXORTrainingData()
        {
            var trainingDataFilePath = $"{Environment.CurrentDirectory}/Resources/XOR_training_data.txt";

            var trainingDataSet = new List<TrainingData<float[]>>();
            var lines = File.ReadAllLines(trainingDataFilePath);

            foreach (var line in lines)
            {
                var splitLine = line.Split(" ");

                var firstInput = float.Parse(splitLine[0]);
                var secondInput = float.Parse(splitLine[1]);
                var target = float.Parse(splitLine[2]);

                var trainingData = new TrainingData<float[]>
                {
                    Data = new[] { firstInput, secondInput },
                    Target = target
                };

                trainingDataSet.Add(trainingData);
            }

            return trainingDataSet;
        }

        static List<TrainingData<float[]>> GetNANDTrainingData()
        {
            var trainingDataFilePath = $"{Environment.CurrentDirectory}/Resources/NAND_training_data.txt";

            var trainingDataSet = new List<TrainingData<float[]>>();
            var lines = File.ReadAllLines(trainingDataFilePath);

            foreach (var line in lines)
            {
                var splitLine = line.Split(" ");

                var firstInput = float.Parse(splitLine[0]);
                var secondInput = float.Parse(splitLine[1]);
                var target = float.Parse(splitLine[2]);

                var trainingData = new TrainingData<float[]>
                {
                    Data = new[] { firstInput, secondInput },
                    Target = target
                };

                trainingDataSet.Add(trainingData);
            }

            return trainingDataSet;
        }

        // Build network with 2 inputs and one output (simple Perceptron)
        //O--
        //   --
        //      -- O
        //   --
        //O--
        static Net BuildNet()
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

            var network = new Net(layers);
            return network;
        }
    }
}
