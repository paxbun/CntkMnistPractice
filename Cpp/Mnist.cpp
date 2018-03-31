#include "Mnist.h"

size_t GetFileSize(std::wstring path)
{
	std::ifstream in(path, std::ifstream::ate | std::ifstream::binary);
	auto rtn = in.tellg();
	in.close();
	return rtn;
}

MnistItem::MnistItem()
	: _Image(nullptr), _Label(0), _Rows(0), _Columns(0)
{
}

MnistItem::MnistItem(void * data, unsigned char label, int rows, int columns)
	: _Label(label), _Image(new unsigned char[rows * columns]), _Rows(rows), _Columns(columns)
{
	memcpy(_Image, data, rows * columns);
}

MnistItem::MnistItem(const MnistItem & other)
	: _Label(other._Label), _Image(new unsigned char[other._Rows * other._Columns]), _Rows(other._Rows), _Columns(other._Columns)
{
	memcpy(_Image, other._Image, _Rows * _Columns);
}

MnistItem::MnistItem(MnistItem && other)
	: _Label(other._Label), _Image(other._Image), _Rows(other._Rows), _Columns(other._Columns)
{
	other._Image = nullptr;
}

MnistItem & MnistItem::operator=(const MnistItem & other)
{
	_Label = other._Label;
	_Image = new unsigned char[other._Rows * other._Columns];
	_Rows = other._Rows;
	_Columns = other._Columns;
	memcpy(_Image, other._Image, _Rows * _Columns);
	return *this;
}

MnistItem & MnistItem::operator=(MnistItem && other)
{
	_Label = other._Label;
	_Image = other._Image;
	_Rows = other._Rows;
	_Columns = other._Columns;
	other._Image = nullptr;
	return *this;
}

MnistItem::~MnistItem()
{
	delete[] _Image;
}

int Mnist::ByteToInt(unsigned char * num)
{
	int result = 0;
	int offset[] = { 24, 16, 8, 0 };
	for (int i = 0; i < 4; i++)
	{
		result |= ((int)num[i]) << offset[i];
	}
	return result;
}

Mnist::Mnist(std::wstring imagePath, std::wstring labelPath, bool normalize)
{
	std::ifstream imageRead(imagePath, std::ifstream::binary);
	std::ifstream labelRead(labelPath, std::ifstream::binary);
	unsigned char * imageTemp = new unsigned char[GetFileSize(imagePath)];
	unsigned char * labelTemp = new unsigned char[GetFileSize(labelPath)];
	imageRead.read((char *)imageTemp, GetFileSize(imagePath));
	labelRead.read((char *)labelTemp, GetFileSize(labelPath));
	int imageMagicNumber = ByteToInt(imageTemp + 0);
	int labelMagicNumber = ByteToInt(labelTemp + 0);
	if (imageMagicNumber != 2051)
	{
		delete[] imageTemp;
		delete[] labelTemp;
		throw std::exception("Not a valid MNIST image data file.");
	}

	if (labelMagicNumber != 2049)
	{
		delete[] imageTemp;
		delete[] labelTemp;
		throw std::exception("Not a valid MNIST label data file.");
	}

	int imageLength = ByteToInt(imageTemp + 4);
	int labelLength = ByteToInt(labelTemp + 4);
	if (imageLength != labelLength)
	{
		delete[] imageTemp;
		delete[] labelTemp;
		throw std::exception("Number of items of two files are not the same.");
	}

	int imageRows = ByteToInt(imageTemp + 8);
	int imageColumns = ByteToInt(imageTemp + 12);

	_Length = imageLength;
	if (normalize)
	{
		_Rows = 1;
		_Columns = imageRows * imageColumns;
	}
	else
	{
		_Rows = imageRows;
		_Columns = imageColumns;
	}
	_Image = new MnistItem[_Length];

	for (int i = 0; i < _Length; i++)
	{
		_Image[i] = MnistItem(imageTemp + 16 + i * _Rows * _Columns, labelTemp[8 + i], _Rows, _Columns);
	}

	delete[] imageTemp;
	delete[] labelTemp;
}

Mnist::~Mnist()
{
	delete[] _Image;
}
