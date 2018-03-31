using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

namespace CntkMnistPractice
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.Write("Loading MNIST models...");
                var directory = ""; // TODO: Add the solution directory so as to read MNIST data files.
                var train = new Mnist(directory + "train-images.idx3-ubyte", directory + "train-labels.idx1-ubyte", true);
                var test = new Mnist(directory + "t10k-images.idx3-ubyte", directory + "t10k-labels.idx1-ubyte", true);
                Console.WriteLine("Done. TrainLength: {0}, TestLength: {1}", train.Length, test.Length);
                
                var device = DeviceDescriptor.GPUDevice(0);
                Console.WriteLine("Using device {0}...", device.AsString());
                
                var startTime = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;

                var classifier = new MnistClassifier(directory + "trainedModel.bin");
                Console.WriteLine("Training...");
                classifier.Train(device, train, true);
                Console.Write("Evaluating...");
                float accuracy = classifier.Evaluate(device, test);
                Console.WriteLine("Done. Accuracy: {0}", accuracy);
                
                Console.WriteLine("Elapsed: {0}", DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond - startTime);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}
