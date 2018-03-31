using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CntkMnistPractice
{
    class Mnist : IEnumerable<MnistItem>
    {
        MnistItem[] image;

        // Returns the number of items
        public int Length { get; }
        // Returns the number of rows of an image.
        public int Rows { get; }
        // Returns the number of columns of an image.
        public int Columns { get; }

        public MnistItem GetAt(int i)
        {
            return image[i];
        }

        private static int ReverseEndian(int num)
        {
            int[] filter = new int[4] { 0x7F000000, 0x00FF0000, 0x0000FF00, 0x000000FF };
            int[] offset = new int[4] { 24, 8, 8, 24 };
            
            int result = 0;
            for (int i = 0; i < 2; i++)
                result |= (num & filter[i]) >> offset[i];
            for (int i = 2; i < 4; i++)
                result |= (num & filter[i]) << offset[i];
            if (num < 0)
                result += 0x00000080;

            return result;
        }

        public Mnist(string imagePath, string labelPath, bool normalize = false)
        {
            byte[] imageTemp = System.IO.File.ReadAllBytes(imagePath);
            byte[] labelTemp = System.IO.File.ReadAllBytes(labelPath);
            int imageMagicNumber = ReverseEndian(BitConverter.ToInt32(imageTemp, 0));
            int labelMagicNumber = ReverseEndian(BitConverter.ToInt32(labelTemp, 0));
            if (imageMagicNumber != 2051)
                throw new Exception("Not a valid MNIST image data file.");

            if (labelMagicNumber != 2049)
                throw new Exception("Not a valid MNIST label data file.");

            int imageLength = ReverseEndian(BitConverter.ToInt32(imageTemp, 4));
            int labelLength = ReverseEndian(BitConverter.ToInt32(labelTemp, 4));
            if (imageLength != labelLength)
                throw new Exception("Number of items of two files are not the same.");

            int imageRows = ReverseEndian(BitConverter.ToInt32(imageTemp, 8));
            int imageColumns = ReverseEndian(BitConverter.ToInt32(imageTemp, 12));

            Length = imageLength;
            if(normalize)
            {
                Rows = 1;
                Columns = imageRows * imageColumns;
            }
            else
            {
                Rows = imageRows;
                Columns = imageColumns;
            }
            image = new MnistItem[Length];
            
            for (int i = 0; i < Length; i++)
            {
                byte[] item = new byte[Rows * Columns];
                Array.Copy(imageTemp, 16 + i * Rows * Columns, item, 0, Rows * Columns);
                image[i] = new MnistItem(item, labelTemp[8 + i], Rows, Columns);
            }

        }
        
        public IEnumerator<MnistItem> GetEnumerator()
        {
            for (int i = 0; i < Length; i++)
            {
                yield return image[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return (IEnumerator)GetEnumerator();
        }
    }

    class MnistItem
    {
        public byte[] Image { get; }
        public byte Label { get; }

        public int Rows { get; }
        public int Columns { get; }

        public MnistItem(byte[] data, byte label, int rows, int columns)
        {
            this.Label = label;
            this.Image = new byte[rows * columns];
            this.Rows = rows;
            this.Columns = columns;
            data.CopyTo(this.Image, 0);
        }
    }

    class NormalizedMnistItem
    {
        public float[] Image { get; }
        public float[] Label { get; }
        public int Rows { get; }
        public int Columns { get; }

        public NormalizedMnistItem(MnistItem item)
        {
            Rows = item.Rows;
            Columns = item.Columns;
            Image = new float[Rows * Columns];
            Label = new float[10];
            Label[item.Label] = 1.0f;
            for (int i = 0; i < item.Image.Length; i++)
            {
                Image[i] = ((float)item.Image[i]) / byte.MaxValue;
            }
        }
    }

    class NormalizedMnistItemDouble
    {
        public double[] Image { get; }
        public double[] Label { get; }
        public int Rows { get; }
        public int Columns { get; }

        public NormalizedMnistItemDouble(MnistItem item)
        {
            Rows = item.Rows;
            Columns = item.Columns;
            Image = new double[Rows * Columns];
            Label = new double[10];
            Label[item.Label - 1] = 1.0f;
            for (int i = 0; i < item.Image.Length; i++)
            {
                Image[i] = ((double)item.Image[i]) / byte.MaxValue;
            }
        }
    }
}
