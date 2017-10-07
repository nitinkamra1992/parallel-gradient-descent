#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>
#include <assert.h>
#include "datalib.h"

using namespace std;

float **x_train, **y_train, **x_test, **y_test, **x_val, **y_val;
const int NUM_CLASSES = 10;
const int NUM_TRAIN = 50000;
const int NUM_TEST = 10000;
const int NUM_VAL = 10000;
const int NUM_FEATURES = 784;

void dataset_mem_alloc()
{
	x_train = (float**) malloc(NUM_TRAIN * sizeof(float*));
	for(int i = 0; i < NUM_TRAIN; ++i)
		x_train[i] = (float*) malloc(NUM_FEATURES * sizeof(float));
	x_val = (float**) malloc(NUM_VAL * sizeof(float*));
	for(int i = 0; i < NUM_VAL; ++i)
		x_val[i] = (float*) malloc(NUM_FEATURES * sizeof(float));
	x_test = (float**) malloc(NUM_TEST * sizeof(float*));
	for(int i = 0; i < NUM_TEST; ++i)
		x_test[i] = (float*) malloc(NUM_FEATURES * sizeof(float));
	y_train = (float**) malloc(NUM_TRAIN * sizeof(float*));
	for(int i = 0; i < NUM_TRAIN; i++)
		y_train[i] = (float*) malloc(NUM_CLASSES * sizeof(float));
	y_val = (float**) malloc(NUM_VAL * sizeof(float*));
	for(int i = 0; i < NUM_VAL; i++)
		y_val[i] = (float*) malloc(NUM_CLASSES * sizeof(float));
	y_test = (float**) malloc(NUM_TEST * sizeof(float*));
	for(int i = 0; i < NUM_TEST; i++)
		y_test[i] = (float*) malloc(NUM_CLASSES * sizeof(float));
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return ((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void readImageFile(string filename, float** arr)
{
	ifstream file(filename.c_str(), ios::binary);
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	if (file.is_open())
	{	
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for(int i = 0; i < number_of_images; ++i)
		{
			for(int j = 0; j < n_rows*n_cols; ++j)
			{
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				arr[i][j] = (float)temp;
			}
		}
	}
	file.close();
}
void readLabelFile(string filename, float** arr)
{
	float* temp_arr;
	int magic_number = 0;
	int number_of_labels = 0;
	ifstream file (filename.c_str(),ios::binary);
	if (file.is_open())
	{		
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_labels, sizeof(number_of_labels));
		number_of_labels = ReverseInt(number_of_labels);
		temp_arr = (float*) malloc(number_of_labels * sizeof(float));

		for(int i = 0; i < number_of_labels; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			temp_arr[i] = (float)temp;
		}
	}
	file.close();

	// Convert to one-hot encoding
	for(int i = 0; i < number_of_labels; i++)
	{
		for(int j = 0; j < NUM_CLASSES; j++)
			arr[i][j] = 0;
		arr[i][(int)temp_arr[i]] = 1;
	}
	free(temp_arr);
}

void readData(bool normalize)
// normalize has default value false
{
	float** X_total;
	float** y_total;

	X_total = (float**) malloc((NUM_TRAIN + NUM_VAL) * sizeof(float*));
	y_total = (float**) malloc((NUM_TRAIN + NUM_VAL) * sizeof(float*));
	for(int i = 0; i < NUM_TRAIN + NUM_VAL; ++i)
	{	
		X_total[i] = (float*) malloc(NUM_FEATURES * sizeof(float));
		y_total[i] = (float*) malloc(NUM_CLASSES * sizeof(float));
	}

	dataset_mem_alloc();

	cout << "Reading files..." << endl;

	readImageFile("../data/train-images.idx3-ubyte",X_total);
	readLabelFile("../data/train-labels.idx1-ubyte",y_total);
	readImageFile("../data/t10k-images.idx3-ubyte",x_test);
	readLabelFile("../data/t10k-labels.idx1-ubyte",y_test);
	
	cout << "Bifurcating data into training and validation sets..." << endl;

	for(int i = 0; i < NUM_TRAIN; i++)
		for(int j = 0; j < NUM_FEATURES; j++)
			x_train[i][j] = X_total[i][j];
	for(int i = 0; i < NUM_VAL; i++)
		for(int j = 0; j < NUM_FEATURES; j++)
			x_val[i][j] = X_total[i+NUM_TRAIN][j];
	for(int i = 0; i < NUM_TRAIN; i++)
		for(int j = 0; j < NUM_CLASSES; j++)
			y_train[i][j] = y_total[i][j];
	for(int i = 0; i < NUM_VAL; i++)
		for(int j = 0; j < NUM_CLASSES; j++)
			y_val[i][j] = y_total[i+NUM_TRAIN][j];

	cout << "Freeing temporary memory..." << endl;

	for(int i = 0; i < NUM_TRAIN + NUM_VAL; ++i)
	{	
		free(X_total[i]);
		free(y_total[i]);
	}
	free(X_total);
	free(y_total);

	// normalize data if needed
	if(normalize)
	{
		for(int j = 0; j < NUM_FEATURES; j++)
		{
			for(int i = 0; i < NUM_TRAIN; i++)
				x_train[i][j] = (x_train[i][j] - 128)/128;
			for(int i = 0; i < NUM_VAL; i++)
				x_val[i][j] = (x_val[i][j] - 128)/128;
			for(int i = 0; i < NUM_TEST; i++)
				x_test[i][j] = (x_test[i][j] - 128)/128;
		}
	}
}