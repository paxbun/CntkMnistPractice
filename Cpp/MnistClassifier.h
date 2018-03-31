#pragma once

#ifndef MNIST_CLASSIFIER_INCLUDED
#define MNIST_CLASSIFIER_INCLUDED

#include "Mnist.h"
#include <string>
#include "CNTKLibrary.h"

using namespace CNTK;
using namespace std;

class MnistClassifier
{
private:
	std::vector<unsigned char> _Model;
	std::wstring _FeatureStreamName = L"features";
	std::wstring _LabelsStreamName = L"labels";
	std::wstring _ClassifierName = L"classifierOutput";

	const int _NumClasses = 10;
	NDShape _ImageDim;
	NDShape _LabelDim;

	std::wstring _ModelPath;

public:
	MnistClassifier(const std::wstring & modelPath);
	void Train(const DeviceDescriptor & device, const Mnist & trainItems, bool ignoreTrainedFile = false);
	float Evaluate(const DeviceDescriptor & device, const Mnist & testItems);
	~MnistClassifier();

private:
	FunctionPtr FullyConnectedLinearLayer(const Variable & input, int outputDim, const DeviceDescriptor & device, const std::wstring & outputName = L"");
};

#endif