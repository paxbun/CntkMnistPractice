#include "CNTKLibrary.h"

#include <iostream>
#include <string>
#include <ctime>
#include "Mnist.h"
#include "MnistClassifier.h"

#define SOLUTION_DIR L""

using namespace std;
using namespace CNTK;

int main(int argc, char * argv[])
{
	try
	{
		wcout << "Loading MNIST models...";
		wstring directory = L""; // TODO: Add the solution directory so as to read MNIST data files.
		Mnist train(directory + L"\\MNIST\\train-images.idx3-ubyte", directory + L"\\MNIST\\train-labels.idx1-ubyte", true);
		Mnist test(directory + L"\\MNIST\\t10k-images.idx3-ubyte", directory + L"\\MNIST\\t10k-labels.idx1-ubyte", true);
		wcout << "Done. TrainLength: " << train.GetLength() << ", TestLength: " << test.GetLength() << endl;

		auto device = DeviceDescriptor::GPUDevice(0);
		wcout << "Using device " << device.AsString() << "..." << endl;

		clock_t startTime = clock();

		MnistClassifier classifier(directory + L"trainedModel.bin");
		wcout << "Training..." << endl;
		classifier.Train(device, train, 3);
		wcout << "Evaluating...";
		float accuracy = classifier.Evaluate(device, test);
		wcout << "Done. Accuracy: " << accuracy << endl;

		wcout << "Elapsed: " << clock() - startTime << endl;
	}
	catch (const std::exception & e)
	{
		wcout << e.what() << endl;
	}
}