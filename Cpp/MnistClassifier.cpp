#include "MnistClassifier.h"
#include <vector>
#include <map>
#include <iostream>

MnistClassifier::MnistClassifier(const std::wstring & modelPath)
	: _ModelPath(modelPath)
{
}

void MnistClassifier::Train(const DeviceDescriptor & device, const Mnist & trainItems, size_t epoches, bool ignoreTrainedFile)
{
	_ImageDim = { (size_t)trainItems.GetRows() * trainItems.GetColumns() };
	_LabelDim = { (size_t)_NumClasses };

	std::ifstream modelRead(_ModelPath, std::ifstream::binary);
	if (modelRead && !ignoreTrainedFile)
	{
		std::wcout << "Trained model exists. Terminating training." << std::endl;
		std::ifstream modelRead(_ModelPath, std::ifstream::binary);
		_Model.resize(GetFileSize(_ModelPath));
		modelRead.read((char*)_Model.data(), _Model.size());
		modelRead.close();
		return;
	}
	else
		modelRead.close();

	auto input = InputVariable(_ImageDim, DataType::Float, _FeatureStreamName);
	auto scaledInput = ElementTimes(Constant::Scalar<float>(0.00390625f, device), input);

	auto layer1 = FullyConnectedLinearLayer(input, 50, device, L"layer1");
	auto layer2 = FullyConnectedLinearLayer(layer1, 50, device, L"layer2");
	
	auto classifierOutput = FullyConnectedLinearLayer(layer2, 10, device, _ClassifierName);

	auto labels = InputVariable(_LabelDim, DataType::Float, _LabelsStreamName);
	auto trainingLoss = CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
	auto prediction = ClassificationError(classifierOutput, labels, L"classificationError");

	TrainingParameterSchedule<double> learningRatePerSample(0.003125, 1);
	std::vector<LearnerPtr> parameterLearners = { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) };
	auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

	for (int epoch = 1; epoch <= epoches; epoch++)
	{
		for (auto& item : trainItems)
		{
			NormalizedMnistItem<float> normalized(item);
			std::unordered_map<Variable, ValuePtr> arguments = 
			{
				{ input, Value::CreateBatch(_ImageDim, normalized.GetImage(), device) },
				{ labels, Value::CreateBatch(_LabelDim, normalized.GetLabel(), device)},
			};
			trainer->TrainMinibatch(arguments, false, device);
		}
	}

	classifierOutput->Save(_Model);
	std::ofstream modelWrite(_ModelPath, std::ofstream::binary);
	modelWrite.write((char *)_Model.data(), _Model.size());
}

float MnistClassifier::Evaluate(const DeviceDescriptor & device, const Mnist & testItems)
{
	auto classifier = Function::Load((char *)_Model.data(), _Model.size(), device);

	auto input = classifier->Arguments()[0];
	auto output = classifier->Output();

	size_t correct = 0;
	for (auto & item : testItems)
	{
		NormalizedMnistItem<float> normalized(item);
		std::unordered_map<Variable, ValuePtr> inputArguments =
		{
			{ input, Value::CreateBatch(_ImageDim, normalized.GetImage(), device) }
		};
		std::unordered_map<Variable, ValuePtr> outputArguments =
		{
			{ output, nullptr }
		};

		classifier->Evaluate(inputArguments, outputArguments, device);
		std::vector<std::vector<float>> outputDataTemp;
		outputArguments.at(output)->CopyVariableValueTo(output, outputDataTemp);
		auto & outputData = outputDataTemp[0];
		int maxIndex = 0;
		float maxValue = outputData[0];
		for (int i = 1; i < 10; i++)
			if (maxValue < outputData[i])
			{
				maxValue = outputData[i];
				maxIndex = i;
			}

		if (maxIndex == item.GetLabel())
			correct++;
	}

	return ((float)correct) / testItems.GetLength();
}

MnistClassifier::~MnistClassifier()
{
}

FunctionPtr MnistClassifier::FullyConnectedLinearLayer(const Variable & input, int outputDim, const DeviceDescriptor & device, const std::wstring & outputName)
{
	size_t inputDim = input.Shape()[0];

	NDShape s = { (size_t)outputDim, (size_t)inputDim };
	Parameter timesParam(
		s, DataType::Float,
		GlorotUniformInitializer(
			DefaultParamInitScale,
			SentinelValueForInferParamInitRank,
			SentinelValueForInferParamInitRank, 1
		),
		device,
		L"timesParam"
	);
	auto timesFunction = Times(timesParam, input, L"times");

	NDShape s2 = { (size_t)outputDim };
	Parameter plusParam(s2, 0.0f, device, L"plusParam");

	return Plus(plusParam, timesFunction, outputName);
}
