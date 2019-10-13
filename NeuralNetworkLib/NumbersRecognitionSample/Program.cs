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

        static void Main(string[] args)
        {
            //Get training data from MNIST
            var imagesTrainingData = GetImages(_imagesPath, _labelsPath);
        }

        private static List<TrainingData<float[]>> GetImages(string imagesPath, string labelsPath)
        {
            var trainingData = new List<TrainingData<float[]>>();

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

                    var trainingDataItem = new TrainingData<float[]>
                    {
                        //255 + 1 in order to make target no 1, but 0.9999.
                        Data = bytes.Select(b => (float)b / (255 + 1)).ToArray(),
                        Target = labels.ReadByte()
                    };

                    trainingData.Add(trainingDataItem);
                }
            }

            return trainingData;
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
