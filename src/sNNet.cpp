#include <fstream>
#include <iomanip>
#include <string>
#include <numeric>
#include <algorithm> 
#include <unistd.h>
#include <pthread.h>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <assert.h>
#include <vector>
#include <sstream>
#include "../common/Time.h"
#include "../common/Constants.h"

#include "matrixop.h"
#include "datalib.h"

using namespace std;

int num_layers;
int *layers_size;
float ***w, **b, **delta, **a, **z, **sigDz;
float *delC_a, ***delC_w, **delC_b;

float lambda = 1e-3;
float alpha = 1e-1;

int miniBatchSize = 128;
int nEpochs = 50;

void allocate_memory()
{

	ifstream f("net.config");
	f >> num_layers;
	layers_size = (int *)malloc( (num_layers) * sizeof(int) );
	for( int i = 0; i < num_layers; i++)
		f >> layers_size[i];
	
	w = (float ***)malloc( (num_layers - 1) * sizeof(float **) ); 
	delC_w = (float ***)malloc( (num_layers - 1) * sizeof(float **) );
	b = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // b idx 0 corresponds to 1st hidden layer
	delta = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // delta idx 0 corresponds to 1st hidden layer
	delC_b = (float **)malloc( (num_layers - 1) * sizeof(float *) );
	a = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // a idx 0 corresponds to 1st hidden layer
	z = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // z idx 0 corresponds to 1st hidden layer
	sigDz = (float **)malloc( (num_layers - 1) * sizeof(float *) );
	for( int i = 0 ; i < num_layers - 1; i++)
	{
		w[i] = (float **)malloc( layers_size[i] * sizeof(float *) );
		delC_w[i] = (float **)malloc( layers_size[i] * sizeof(float *) );
		b[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		delta[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		delC_b[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		a[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		z[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		sigDz[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		for( int j = 0; j < layers_size[i]; j++)
		{
			w[i][j] = (float *)malloc( layers_size[i+1] * sizeof(float) );
			delC_w[i][j] = (float *)malloc( layers_size[i+1] * sizeof(float) );
		}
	}
	delC_a = (float *)malloc( layers_size[num_layers - 1] * sizeof(float) );
	assert(delC_a != NULL);
}

void initializeGlorot()
{
	unsigned long seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_real_distribution<float> distribution(-1.0,1.0);
	
	for(int i = 0; i < num_layers - 1; i++)
	{
		float glorotConstant = sqrt(6)/sqrt(layers_size[i + 1] + layers_size[i]);
		for(int j = 0; j < layers_size[i + 1]; j++)
		{
			for(int k = 0; k < layers_size[i]; k++)
				w[i][k][j] = glorotConstant*distribution(generator);
			b[i][j] = 0;
		}
	}
}

void forwardPass(int idx, int dataset_type)// Idx is the sample idx in the data set
{
	Time<secs> Timer;
	Timer.reset();
	switch(dataset_type)
	{
		case 1: mvProdT(w[0], x_train[idx], z[0], layers_size[0], layers_size[1]);
				break;
		case 2: mvProdT(w[0], x_val[idx], z[0], layers_size[0], layers_size[1]);
				break;
		case 3: mvProdT(w[0], x_test[idx], z[0], layers_size[0], layers_size[1]);
				break;
	}
	add(z[0], b[0], z[0], layers_size[1]);
	sigmoid(z[0], a[0], layers_size[1]);
	dSigmoid(z[0], sigDz[0], layers_size[1]);
	int i;
	for( i = 1; i < num_layers - 1; i++)
	{
		mvProdT(w[i], a[i - 1], z[i], layers_size[i], layers_size[i+1]);
		add(z[i], b[i], z[i], layers_size[i+1]);
		sigmoid(z[i], a[i], layers_size[i+1]);
		dSigmoid(z[i], sigDz[i], layers_size[i+1]);
	}
//	Timer.lap("secs");
	// Add the softmax layer
	// mvProdT(w[i], a[i - 1], z[i], layers_size[i], layers_size[i+1]);
	// add(z[i], b[i], z[i], layers_size[i+1]);
	// softmax(z[i], a[i], layers_size[i+1]);
	// softmaxD(z[i], sigDz[i], layers_size[i+1]);
}

void backwardPass(int idx)
{
	Time<secs> Timer;
	Timer.reset();
	costFnLMSD(y_train[idx], a[num_layers - 2], delC_a, layers_size[num_layers - 1]);
	hprod(delC_a, sigDz[num_layers - 2], delta[num_layers - 2], layers_size[num_layers - 1]);
	add(delC_b[num_layers - 2], delta[num_layers - 2], delC_b[num_layers - 2], layers_size[num_layers - 1]);
	for( int j = 0; j < layers_size[num_layers - 2]; j++)
		for( int k = 0; k < layers_size[num_layers - 1]; k++)
			delC_w[num_layers - 2][j][k] += ((num_layers - 2 > 0) ? a[num_layers - 3][j] : x_train[idx][j])*delta[num_layers - 2][k] ;//+ 2*lambda*w[num_layers - 2][j][k];
	for(int i = num_layers - 3; i >= 0; i--)
	{
		float *temp = (float *)malloc(layers_size[i+1] * sizeof(float));
		mvProd(w[i+1], delta[i+1], temp, layers_size[i+1], layers_size[i+2]);
		hprod(temp, sigDz[i], delta[i], layers_size[i+1]);
		add(delC_b[i], delta[i], delC_b[i], layers_size[i+1]);
		for( int j = 0; j < layers_size[i]; j++)
			for( int k = 0; k < layers_size[i+1]; k++)
				delC_w[i][j][k] += ((i > 0) ? a[i-1][j] : x_train[idx][j])*delta[i][k] ;//+ 2*lambda*w[i][j][k];
	}
	// Timer.lap("secs");
}

void initDeriv()
{
	Time<secs> Timer;
	Timer.reset();
	for(int i = 0; i < num_layers - 1; i++)
		for(int j = 0; j < layers_size[i+1]; j++)
			delC_b[i][j] = 0;
	for(int i = 0; i < num_layers - 1; i++)
		for(int j = 0; j < layers_size[i]; j++)
			for(int k = 0; k < layers_size[i+1]; k++)
				delC_w[i][j][k] = 0;
	// Timer.lap("secs");
}

void updateStochastic(int idx, int dataset_type)
{
	initDeriv();
	forwardPass(idx, dataset_type);
	backwardPass(idx);
	for(int i = 0; i < num_layers - 1; i++)
	{
		prod(-alpha, delC_b[i], delC_b[i], layers_size[i+1]);
		add(b[i], delC_b[i], b[i], layers_size[i+1]);
	}
	for(int i = 0; i < num_layers - 1; i++)
	{
		for(int j = 0; j < layers_size[i]; j++)
		{
			prod(-alpha, delC_w[i][j], delC_w[i][j], layers_size[i+1]);
			add(w[i][j], delC_w[i][j], w[i][j], layers_size[i+1]);
		}
	}
}

void updateMiniBatch(int start_idx, int end_idx, int dataset_type)
{
	initDeriv();
	for(int i = start_idx; i <= end_idx; i++)
	{
		forwardPass(i, dataset_type);
		backwardPass(i);
	}
	for(int i = 0; i < num_layers - 1; i++)
	{
		prod(-(alpha/(end_idx - start_idx + 1)), delC_b[i], delC_b[i], layers_size[i+1]);///(end_idx - start_idx + 1)
		add(b[i], delC_b[i], b[i], layers_size[i+1]);
	}
	for(int i = 0; i < num_layers - 1; i++)
	{
		for(int j = 0; j < layers_size[i]; j++)
		{
			prod(-(alpha/(end_idx - start_idx + 1)), delC_w[i][j], delC_w[i][j], layers_size[i+1]);
			add(w[i][j], delC_w[i][j], w[i][j], layers_size[i+1]);
		}
	}
}

void trainStochastic(int idx)
{
	updateStochastic(idx, 1);
}

void trainMiniBatch(int start_idx, int end_idx)
{
	updateMiniBatch(start_idx, end_idx, 1);
}

int testAccuracy(int idx, int dataset_type)
{
	forwardPass(idx, dataset_type);
	switch(dataset_type)
	{
		case 1: if(equals(a[num_layers - 2], y_train[idx], layers_size[num_layers - 1]))
					return 1;
				break;
		case 2: if(equals(a[num_layers - 2], y_val[idx], layers_size[num_layers - 1]))
					return 1;
				break;
		case 3: if(equals(a[num_layers - 2], y_test[idx], layers_size[num_layers - 1]))
					return 1;
				break;
	}
	
	return 0;
}

float testCost(int idx, int dataset_type)
{
	forwardPass(idx, dataset_type);
	switch(dataset_type)
	{
		case 1:	
				return costFnLMS(y_train[idx], a[num_layers - 2], layers_size[num_layers - 1]);
		case 2: 
				return costFnLMS(y_val[idx], a[num_layers - 2], layers_size[num_layers - 1]);
		case 3: 
				return costFnLMS(y_test[idx], a[num_layers - 2], layers_size[num_layers - 1]);
	}
	
	return 0;
}

float testBatchAccuracy(int start_idx, int end_idx, int dataset_type)
{
	float accuracy = 0;
	for(int i = start_idx; i <= end_idx; i++)
		accuracy += testAccuracy(i, dataset_type);
	accuracy /= (end_idx - start_idx + 1);
	return accuracy;
}

float testBatchCost(int start_idx, int end_idx, int dataset_type)
{
	float entr = 0;
	for(int i = start_idx; i <= end_idx; i++)
		entr += testCost(i, dataset_type);
	entr /= (end_idx - start_idx + 1);
	return entr;
}

void train()
{
	Time<secs> Timer;
	int numMiniBatches = NUM_TRAIN/miniBatchSize;
	float accuracy;
	float entr;
	initializeGlorot();
//	ofstream fout("Cost_train.log");
	double timePerEpoch = 0.0;
	for(int epoch = 0; epoch < nEpochs; epoch++)
	{
		Timer.reset();
		int i;
		for(i = 0; i < numMiniBatches - 1; i++)
			trainMiniBatch(i*miniBatchSize, (i+1)*miniBatchSize - 1);
		trainMiniBatch(i*miniBatchSize, NUM_TRAIN - 1);
		cout << "Epoch: " << epoch << endl;
		if(epoch > 0)
		{
			cout << "\t\t"; 
			timePerEpoch += Timer.lap("secs");
		}
//		entr = 	testBatchCost(0, NUM_TRAIN - 1, 1);			
//		cout << "\t\tTraining cost = " << entr << endl;		
//		fout << entr << endl;
		accuracy = testBatchAccuracy(0, NUM_VAL - 1, 2);
		cout << "\t\tValidation accuracy = " << accuracy*100 << "%" << endl;
	}
	timePerEpoch /= (nEpochs - 1);
	cout << "Average time per epoch = " << timePerEpoch << endl;
//	fout.close();
}

int main(int argc, char *argv[])
{
	allocate_memory();
	readData(false);
	
	train();
	// float** M = (float**) malloc(3 * sizeof(float*));
	// for(int i = 0; i < 3; ++i)
	// {
	// 	M[i] = (float*) malloc(2 * sizeof(float));
	// 	for(int j = 0; j < 2; j++)
	// 		M[i][j] = i+j;
	// }
	// float* x = (float*) malloc(3 * sizeof(float));
	// for(int i = 0; i < 3; i++)
	// 	x[i] = 2-i;
	// float* y = (float*) malloc(2 * sizeof(float));
	// mvProdT(M, x, y, 3, 2);
	// cout << y[0] << " " << y[1] << endl;
}

