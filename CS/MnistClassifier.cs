using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

namespace CntkMnistPractice
{
    class MnistClassifier
    {

        private byte[] model;

        private string featureStreamName = "features";
        private string labelsStreamName = "labels";
        private string classifierName = "classifierOutput";

        private int numClasses = 10;
        private int[] imageDim;
        private int[] labelDim;

        private string modelPath;

        public MnistClassifier(string modelPath)
        {
            this.modelPath = modelPath;
        }

        public void Train(DeviceDescriptor device, Mnist trainItems, int epoches = 1, bool ignoreTrainedFile = false)
        {
            imageDim = new int[] { trainItems.Rows * trainItems.Columns };
            labelDim = new int[] { numClasses };

            if (System.IO.File.Exists(modelPath) && !ignoreTrainedFile)
            {
                Console.WriteLine("Trained model exists. Terminating training.");
                model = System.IO.File.ReadAllBytes(modelPath);
                return;
            }

            var input = CNTKLib.InputVariable(imageDim, DataType.Float, featureStreamName);
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);

            var layer1 = FullyConnectedLinearLayer(input, 50, device, "layer1");
            var layer2 = FullyConnectedLinearLayer(layer1, 100, device, "layer2");

            var classifierOutput = FullyConnectedLinearLayer(layer2, 10, device, classifierName);

            var labels = CNTKLib.InputVariable(labelDim, DataType.Float, labelsStreamName);
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(classifierOutput, labels, "classificationError");
            
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.003125, 1);
            var parameterLearners = new List<Learner>() { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            for (int epoch = 1; epoch <= 1; epoch++)
            {
                foreach (var item in trainItems)
                {
                    var normalized = new NormalizedMnistItem(item);
                    var arguments = new Dictionary<Variable, Value>
                    {
                        //{ input, new Value(new NDArrayView(imageDim, normalized.Image, device)) },
                        { input, Value.CreateBatch(imageDim, normalized.Image, device) },
                        { labels, Value.CreateBatch(labelDim, normalized.Label, device) },
                    };
                    trainer.TrainMinibatch(arguments, false, device);
                }
            }
            
            model = classifierOutput.Save();
            System.IO.File.WriteAllBytes(modelPath, model);
        }

        public float Evaluate(DeviceDescriptor device, Mnist testItems)
        {
            var classifier = Function.Load(model, device);

            var input = classifier.Arguments[0];
            var output = classifier.Output;

            var correct = 0;

            foreach (var item in testItems)
            {
                var normalized = new NormalizedMnistItem(item);
                var inputArguments = new Dictionary<Variable, Value>
                {
                    { input, Value.CreateBatch(imageDim, normalized.Image, device) },
                };
                var outputArguments = new Dictionary<Variable, Value>
                {
                    { output, null }
                };

                classifier.Evaluate(inputArguments, outputArguments, device);
                var outputData = outputArguments[output].GetDenseData<float>(output)[0];
                var maxIndex = 0;
                float maxValue = outputData[0];
                for (int i = 1; i < 10; i++)
                    if(maxValue < outputData[i])
                    {
                        maxValue = outputData[i];
                        maxIndex = i;
                    }

                if (maxIndex == item.Label)
                    correct++;
            }

            return ((float)correct) / testItems.Length;
        }

        private static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device, string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];

            int[] s = { outputDim, inputDim };
            var timesParam = new Parameter(s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");

            int[] s2 = { outputDim };
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            return CNTKLib.ReLU(CNTKLib.Plus(plusParam, timesFunction, outputName));
        }
    }
}
