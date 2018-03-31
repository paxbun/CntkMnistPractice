
# CntkMnistPractice
MNIST classifier using CNTK written in C++ and C#. Only used fully connected layers.
There are two versions of this program - both ones do the exactly same thing.

## Files
### Program.cpp / Program.cs
The entry point of the program.

### Mnist.cpp (Mnist.h) / Mnist.cs
The file has the definition of Mnist class and MnistItem, NormalizedMnistItem.
#### Mnist class
The list of Mnist items.
#### MnistItem class
The class incldues image information and label information of single MNIST item.
#### NormalizedMnistItem class
The class converts values of range [0, 255] to values of range [0.0, 1.0] and does One-hot encoding.

### MnistClassifier.cpp (MnistClassifier.h) / MnistClassifier.cs
The file has the definition of MnistClassifier class, which does training and evaluation.

## etc
The owner of this repository is neither a experienced programmer nor a skilled English speaker; there might be some foolish programmatical errors and some English errors. Please inform me if there is any kind of error in files of this repository.
