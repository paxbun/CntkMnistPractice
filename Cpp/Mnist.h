#pragma once

#ifndef MNIST_INCLUDED
#define MNIST_INCLUDED

#include <cstring>
#include <fstream>
#include <string>
#include <limits>
#include <vector>

size_t GetFileSize(std::wstring path);

class MnistItem
{
private:
	unsigned char * _Image;
	unsigned char _Label;
	int _Rows;
	int _Columns;

public:
	MnistItem();
	MnistItem(void * data, unsigned char label, int rows, int columns);
	MnistItem(const MnistItem & other);
	MnistItem(MnistItem && other);
	MnistItem & operator=(const MnistItem & other);
	MnistItem & operator=(MnistItem && other);
	~MnistItem();

public:
	inline unsigned char * GetImage() { return _Image; } 
	inline const unsigned char * GetImage() const { return _Image; }
	inline unsigned char GetLabel() const { return _Label; }
	inline int GetRows() const { return _Rows; }
	inline int GetColumns() const { return _Columns; }
};

template<class FloatType>
class NormalizedMnistItem
{
private:
	std::vector<FloatType> _Image;
	std::vector<FloatType> _Label;
	int _Rows;
	int _Columns;

public:
	NormalizedMnistItem(const MnistItem & item)
		: _Rows(item.GetRows()), _Columns(item.GetColumns()), _Image(item.GetRows() * item.GetColumns(), 0), _Label(10, 0)
	{
		_Label[item.GetLabel()] = static_cast<FloatType>(1.0);
		for (int i = 0; i < _Rows * _Columns; i++)
		{
			_Image[i] = ((FloatType)item.GetImage()[i]) / std::numeric_limits<unsigned char>::max();
		}
	}
	~NormalizedMnistItem()
	{
	}

public:
	inline std::vector<FloatType> & GetImage() { return _Image; }
	inline std::vector<FloatType> & GetLabel() { return _Label; }
	inline const std::vector<FloatType> & GetImage() const { return _Image; }
	inline const std::vector<FloatType> & GetLabel() const { return _Label; }
	inline int GetRows() const { return _Rows; }
	inline int GetColumns() const { return _Columns; }
};

class Mnist
{
private:
	MnistItem* _Image;
	int _Length;
	int _Rows;
	int _Columns;

public:
	int GetLength() const { return _Length; }
	int GetRows() const { return _Rows; }
	int GetColumns() const { return _Columns; }

public:
	inline MnistItem & GetAt(int i)
	{
		return _Image[i];
	}

	inline const MnistItem & GetAt(int i) const
	{
		return _Image[i];
	}

private:
	static int ByteToInt(unsigned char * num);

public:
	Mnist(std::wstring imagePath, std::wstring labelPath, bool normalize = false);

	~Mnist();

	inline MnistItem * begin()
	{
		return _Image;
	}

	inline const MnistItem * begin() const
	{
		return _Image;
	}

	inline MnistItem * end()
	{
		return _Image + _Length;
	}

	inline const MnistItem * end() const
	{
		return _Image + _Length;
	}
};

#endif