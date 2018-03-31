#include "Mnist.h"
#include <cstdio>

size_t GetFileSize(const std::wstring & path)
{
	struct _stat64 st;
	if (_wstat64(path.c_str(), &st) != 0) {
		return 0;
	}
	return st.st_size;
}

MnistItem::MnistItem()
	: _Image(nullptr), _Label(0), _Rows(0), _Columns(0)
{
}

MnistItem::MnistItem(void * data, unsigned char label, size_t rows, size_t columns)
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

size_t Mnist::ByteToInt(unsigned char * num)
{
	size_t result = 0;
	size_t offset[] = { 24, 16, 8, 0 };
	for (size_t i = 0; i < 4; i++)
	{
		result |= ((size_t)num[i]) << offset[i];
	}
	return result;
}

Mnist::Mnist(std::wstring imagePath, std::wstring labelPath, bool normalize)
{
	std::ifstream imageRead(imagePath, std::ifstream::binary);
	std::ifstream labelRead(labelPath, std::ifstream::binary);
	if (!labelRead || !imageRead)
		throw std::exception("File doesn't exsist.");
	unsigned char * imageTemp = new unsigned char[GetFileSize(imagePath)];
	unsigned char * labelTemp = new unsigned char[GetFileSize(labelPath)];
	imageRead.read((char *)imageTemp, GetFileSize(imagePath));
	labelRead.read((char *)labelTemp, GetFileSize(labelPath));
	imageRead.close();
	labelRead.close();
	size_t imageMagicNumber = ByteToInt(imageTemp + 0);
	size_t labelMagicNumber = ByteToInt(labelTemp + 0);
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

	size_t imageLength = ByteToInt(imageTemp + 4);
	size_t labelLength = ByteToInt(labelTemp + 4);
	if (imageLength != labelLength)
	{
		delete[] imageTemp;
		delete[] labelTemp;
		throw std::exception("Number of items of two files are not the same.");
	}

	size_t imageRows = ByteToInt(imageTemp + 8);
	size_t imageColumns = ByteToInt(imageTemp + 12);

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

	for (size_t i = 0; i < _Length; i++)
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
