#include "datalib.h"
#include <fstream>
#include <iostream>

using namespace std;

void convertRowMajor()
{
	readData(false);
	ofstream fout;

	fout.open("../data/mnist_train.csv", ios::binary);
	cout << "Converting training data to CSV" << endl;
	for(int i = 0; i < NUM_TRAIN; i++)
	{
		for(int j = 0; j < NUM_FEATURES; j++)
		{
			fout << x_train[i][j] << ", ";
		}
		for(int j = 0; j < NUM_CLASSES; j++)
		{
			fout << y_train[i][j];
			if(j < NUM_CLASSES - 1)
				fout << ", ";
		}
		if(i < NUM_TRAIN - 1)
			fout << endl;
	}
	fout.close();

	fout.open("../data/mnist_test.csv", ios::binary);
	cout << "Converting test data to CSV" << endl;
	for(int i = 0; i < NUM_TEST; i++)
	{
		for(int j = 0; j < NUM_FEATURES; j++)
		{
			fout << x_test[i][j] << ", ";
		}
		for(int j = 0; j < NUM_CLASSES; j++)
		{
			fout << y_test[i][j];
			if(j < NUM_CLASSES - 1)
				fout << ", ";
		}
		if(i < NUM_TEST - 1)
			fout << endl;
	}
	fout.close();

	fout.open("../data/mnist_val.csv", ios::binary);
	cout << "Converting validation data to CSV" << endl;
	for(int i = 0; i < NUM_VAL; i++)
	{
		for(int j = 0; j < NUM_FEATURES; j++)
		{
			fout << x_val[i][j] << ", ";
		}
		for(int j = 0; j < NUM_CLASSES; j++)
		{
			fout << y_val[i][j];
			if(j < NUM_CLASSES - 1)
				fout << ", ";
		}
		if(i < NUM_VAL - 1)
			fout << endl;
	}
	fout.close();	
}

int main()
{
	convertRowMajor();
	return 0;
}