using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Mime;
using NeuralNetworkLib;

namespace NumbersRecognitionSample
{
    class Program
    {
        private static string _imagesPath = $"{Environment.CurrentDirectory}/Resources/train-images.idx3-ubyte";
        private static string _labelsPath = $"{Environment.CurrentDirectory}/Resources/train-labels.idx1-ubyte";

        private static float _learningRate = 0.4f;

        static void Main()
        {
            var net = BuildNetwork();

            for (var epochIndex = 0; epochIndex < 500; epochIndex++)
            {
                //Get training data from MNIST
                var imagesTrainingData = GetTrainingData(_imagesPath, _labelsPath);

                foreach (var trainingData in imagesTrainingData)
                {
                    net.FeedForward(trainingData.Data);
                    net.CalculateError(trainingData.Targets);
                    net.BackPropagate(trainingData.Targets, _learningRate);

                    var response = net.GetResult();
                }
            }
        }

        private static MultipleOutputNet BuildNetwork()
        {
            var inputLayer = GenerateLayer(28 * 28);
            var hiddenLayer = GenerateLayer(100);
            var outputLayer = GenerateLayer(10);

            var layers = new List<NeuronLayer> { inputLayer, hiddenLayer, outputLayer };
            
            var net = new MultipleOutputNet(layers);
            return net;
        }

        private static NeuronLayer GenerateLayer(int count)
        {
            var neurons = new List<Neuron>();

            for (var neuronIndex = 0; neuronIndex < count; neuronIndex++)
                neurons.Add(new Neuron());

            //add bias
            neurons.Add(new Neuron());

            var neuronLayer = new NeuronLayer
            {
                Neurons = neurons
            };

            return neuronLayer;
        }

        private static List<TrainingData<float[], float[]>> GetTrainingData(string imagesPath, string labelsPath)
        {
            var trainingData = new List<TrainingData<float[], float[]>>();

            using (var labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open)))
            using (var images = new BinaryReader(new FileStream(imagesPath, FileMode.Open)))
            {
                //need to read some unnecessary values because of file format.
                var magicNumber = ReadBigInt32(images);
                var numberOfImages = ReadBigInt32(images);
                var width = ReadBigInt32(images);
                var height = ReadBigInt32(images);

                var magicLabel = ReadBigInt32(labels);
                var numberOfLabels = ReadBigInt32(labels);

                for (var imageIndex = 0; imageIndex < numberOfImages; imageIndex++)
                {
                    var bytes = images.ReadBytes(width * height);

                    var trainingDataItem = new TrainingData<float[], float[]>
                    {
                        //255 + 1 in order to make target no 1, but 0.9999.
                        Data = bytes.Select(b => (float)b / (255 + 1)).ToArray(),
                        Targets = ConvertNumberToTargetValues(labels.ReadByte())
                    };

                    trainingData.Add(trainingDataItem);
                }
            }

            return trainingData;
        }

        private static float[] ConvertNumberToTargetValues(byte label)
        {
            var targetArr = new float[10];

            for (var targetIndex = 0; targetIndex < 10; targetIndex++)
                targetArr[targetIndex] = 0.0001f;

            switch (label)
            {
                case 1: targetArr[0] = 0.9999f;
                    break;
                case 2:
                    targetArr[1] = 0.9999f;
                    break;
                case 3:
                    targetArr[2] = 0.9999f;
                    break;
                case 4:
                    targetArr[3] = 0.9999f;
                    break;
                case 5:
                    targetArr[4] = 0.9999f;
                    break;
                case 6:
                    targetArr[5] = 0.9999f;
                    break;
                case 7:
                    targetArr[6] = 0.9999f;
                    break;
                case 8:
                    targetArr[7] = 0.9999f;
                    break;
                case 9:
                    targetArr[8] = 0.9999f;
                    break;
                case 10:
                    targetArr[9] = 0.9999f;
                    break;
            }

            return targetArr;
        }

        private static int ReadBigInt32(BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(int));

            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);

            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
