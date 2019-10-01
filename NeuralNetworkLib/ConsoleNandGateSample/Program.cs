using System;
using System.Collections.Generic;
using System.IO;
using NeuralNetworkLib;

namespace ConsoleNandGateSample
{
    class Program
    {
        static void Main()
        {
            var net = BuildNet();
            var trainingDataSet = GetTrainingData();

            foreach (var trainingData in trainingDataSet)
            {
                var response = net.FeedForward(trainingData.Data);

                var error = Math.Abs(response - trainingData.Target);
            }
        }

        static List<TrainingData<float[]>> GetTrainingData()
        {
            var trainingDataFilePath = $"{Environment.CurrentDirectory}/Resources/training_data.txt";

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
                    new Neuron()
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
            layers.Add(outputLayer);

            var network = new Net(layers);
            return network;
        }
    }
}
