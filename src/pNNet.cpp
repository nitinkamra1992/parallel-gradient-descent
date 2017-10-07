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

#include "matrixop.h"
#include "datalib.h"
#include "../common/Time.h"
#include "../common/Constants.h"

#define NUM_THREADS 32

using namespace std;

int num_layers;
int *layers_size;
float ***w, **b, ***delta, ***a, ***z, ***sigDz;
float **delC_a, ****delC_w, ***delC_b;

float lambda = 1e-3;
float alpha = 1e-1;

int miniBatchSize = 128;
int nEpochs = 50;
float accur[NUM_THREADS];


typedef struct my_param
{
	int procNo;
	int startId;
	int endId;
	int dataset_type;
}thread_params;

void allocate_memory()
{

	ifstream f("net.config");
	f >> num_layers;
	layers_size = (int *)malloc( (num_layers) * sizeof(int) );
	for( int i = 0; i < num_layers; i++)
		f >> layers_size[i];
	
	w = (float ***)malloc( (num_layers - 1) * sizeof(float **) ); 
	b = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // b idx 0 corresponds to 1st hidden layer
	for( int i = 0 ; i < num_layers - 1; i++)
	{
		w[i] = (float **)malloc( layers_size[i] * sizeof(float *) );
		b[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		for( int j = 0; j < layers_size[i]; j++)
			w[i][j] = (float *)malloc( layers_size[i+1] * sizeof(float) );
	}
	
	delC_w = (float ****)malloc( NUM_THREADS * sizeof(float ***) );
	delta = (float ***)malloc( NUM_THREADS * sizeof(float **) ); 
	delC_b = (float ***)malloc( NUM_THREADS * sizeof(float **) );
	a = (float ***)malloc( NUM_THREADS * sizeof(float **) ); 
	z = (float ***)malloc( NUM_THREADS * sizeof(float **) ); 
	sigDz = (float ***)malloc( NUM_THREADS * sizeof(float **) );
	delC_a = (float **)malloc( NUM_THREADS * sizeof(float *) );
	for( int t = 0; t < NUM_THREADS; t++)
	{
		delC_w[t] = (float ***)malloc( (num_layers - 1) * sizeof(float **) );
		delta[t] = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // delta idx 0 corresponds to 1st hidden layer
		delC_b[t] = (float **)malloc( (num_layers - 1) * sizeof(float *) );
		a[t] = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // a idx 0 corresponds to 1st hidden layer
		z[t] = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // z idx 0 corresponds to 1st hidden layer
		sigDz[t] = (float **)malloc( (num_layers - 1) * sizeof(float *) );	
		for( int i = 0 ; i < num_layers - 1; i++)
		{
			
			delC_w[t][i] = (float **)malloc( layers_size[i] * sizeof(float *) );
			
			delta[t][i] = (float *)malloc( layers_size[i+1] * sizeof(float));
			delC_b[t][i] = (float *)malloc( layers_size[i+1] * sizeof(float));
			a[t][i] = (float *)malloc( layers_size[i+1] * sizeof(float));
			z[t][i] = (float *)malloc( layers_size[i+1] * sizeof(float));
			sigDz[t][i] = (float *)malloc( layers_size[i+1] * sizeof(float));
			for( int j = 0; j < layers_size[i]; j++)
				delC_w[t][i][j] = (float *)malloc( layers_size[i+1] * sizeof(float) );
		}
		delC_a[t] = (float *)malloc( layers_size[num_layers - 1] * sizeof(float) );
		assert(delC_a[t] != NULL);
	}
	cout << "Allocated memory to variables..." << endl;
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

void forwardPass(int idx, int dataset_type, int threadId)// Idx is the sample idx in the data set
{
	switch(dataset_type)
	{
		case 1: mvProdT(w[0], x_train[idx], z[threadId][0], layers_size[0], layers_size[1]);
				break;
		case 2: mvProdT(w[0], x_val[idx], z[threadId][0], layers_size[0], layers_size[1]);
				break;
		case 3: mvProdT(w[0], x_test[idx], z[threadId][0], layers_size[0], layers_size[1]);
				break;
	}
	add(z[threadId][0], b[0], z[threadId][0], layers_size[1]);
	sigmoid(z[threadId][0], a[threadId][0], layers_size[1]);
	dSigmoid(z[threadId][0], sigDz[threadId][0], layers_size[1]);
	int i;
	for( i = 1; i < num_layers - 1; i++)
	{
		mvProdT(w[i], a[threadId][i - 1], z[threadId][i], layers_size[i], layers_size[i+1]);
		add(z[threadId][i], b[i], z[threadId][i], layers_size[i+1]);
		sigmoid(z[threadId][i], a[threadId][i], layers_size[i+1]);
		dSigmoid(z[threadId][i], sigDz[threadId][i], layers_size[i+1]);
	}
	// Add the softmax layer
	// mvProdT(w[i], a[i - 1], z[i], layers_size[i], layers_size[i+1]);
	// add(z[i], b[i], z[i], layers_size[i+1]);
	// softmax(z[i], a[i], layers_size[i+1]);
	// softmaxD(z[i], sigDz[i], layers_size[i+1]);
}

void backwardPass(int idx, int threadId)
{
	costFnLMSD(y_train[idx], a[threadId][num_layers - 2], delC_a[threadId], layers_size[num_layers - 1]);
	hprod(delC_a[threadId], sigDz[threadId][num_layers - 2], delta[threadId][num_layers - 2], layers_size[num_layers - 1]);
	add(delC_b[threadId][num_layers - 2], delta[threadId][num_layers - 2], delC_b[threadId][num_layers - 2], layers_size[num_layers - 1]);
	for( int j = 0; j < layers_size[num_layers - 2]; j++)
		for( int k = 0; k < layers_size[num_layers - 1]; k++)
			delC_w[threadId][num_layers - 2][j][k] += ((num_layers - 2 > 0) ? a[threadId][num_layers - 3][j] : x_train[idx][j])*delta[threadId][num_layers - 2][k] ;//+ 2*lambda*w[num_layers - 2][j][k];
	for(int i = num_layers - 3; i >= 0; i--)
	{
		float *temp = (float *)malloc(layers_size[i+1] * sizeof(float));
		mvProd(w[i+1], delta[threadId][i+1], temp, layers_size[i+1], layers_size[i+2]);
		hprod(temp, sigDz[threadId][i], delta[threadId][i], layers_size[i+1]);
		add(delC_b[threadId][i], delta[threadId][i], delC_b[threadId][i], layers_size[i+1]);
		for( int j = 0; j < layers_size[i]; j++)
			for( int k = 0; k < layers_size[i+1]; k++)
				delC_w[threadId][i][j][k] += ((i > 0) ? a[threadId][i-1][j] : x_train[idx][j])*delta[threadId][i][k] ;//+ 2*lambda*w[i][j][k];
	}
}

void initDeriv(int threadId)
{
	for(int i = 0; i < num_layers - 1; i++)
		for(int j = 0; j < layers_size[i+1]; j++)
			delC_b[threadId][i][j] = 0;
	for(int i = 0; i < num_layers - 1; i++)
		for(int j = 0; j < layers_size[i]; j++)
			for(int k = 0; k < layers_size[i+1]; k++)
				delC_w[threadId][i][j][k] = 0;
}

void *miniBatchForBack(void *arg)
{
	int procNo = (*(thread_params*) arg).procNo;
	int startId = (*(thread_params*) arg).startId;
	int endId = (*(thread_params*) arg).endId;
	int dataset_type = (*(thread_params*) arg).dataset_type;
	initDeriv(procNo);
	for(int i = startId; i <= endId; i++)
	{
		forwardPass(i, dataset_type, procNo);
		backwardPass(i, procNo);
	}
}

void updateMiniBatch(int start_idx, int end_idx, int dataset_type)
{
	pthread_t thd[NUM_THREADS];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	int ex_per_proc = (end_idx - start_idx + 1)/NUM_THREADS;
	for(unsigned procNo = 0; procNo < NUM_THREADS; procNo++)
	{	
		thread_params *curr_thread_params = (thread_params *)malloc(sizeof(thread_params));
		(*curr_thread_params).procNo = procNo;
		(*curr_thread_params).startId = start_idx + procNo*ex_per_proc;
		if(procNo != NUM_THREADS - 1)
			(*curr_thread_params).endId = start_idx + (procNo + 1)*ex_per_proc - 1;
		else
			(*curr_thread_params).endId = end_idx;
		(*curr_thread_params).dataset_type = dataset_type;
		pthread_create(&thd[procNo], &attr, miniBatchForBack, (void *)curr_thread_params);
	}
	for(unsigned procNo = 0; procNo < NUM_THREADS; procNo++)
	{
		pthread_join(thd[procNo],NULL);
	}
	
	
	for(int t = 1; t < NUM_THREADS; t++)
		for(int i = 0; i < num_layers - 1; i++)
			add(delC_b[0][i], delC_b[t][i], delC_b[0][i], layers_size[i+1]);
			
	for(int i = 0; i < num_layers - 1; i++)
	{
		prod(-(alpha/(end_idx - start_idx + 1)), delC_b[0][i], delC_b[0][i], layers_size[i+1]);///(end_idx - start_idx + 1)
		add(b[i], delC_b[0][i], b[i], layers_size[i+1]);
	}
	
	for(int t = 1; t < NUM_THREADS; t++)
		for(int i = 0; i < num_layers - 1; i++)
			for(int j = 0; j < layers_size[i]; j++)
				add(delC_w[0][i][j], delC_w[t][i][j], delC_w[0][i][j], layers_size[i+1]);
	for(int i = 0; i < num_layers - 1; i++)
	{
		for(int j = 0; j < layers_size[i]; j++)
		{
			prod(-(alpha/(end_idx - start_idx + 1)), delC_w[0][i][j], delC_w[0][i][j], layers_size[i+1]);
			add(w[i][j], delC_w[0][i][j], w[i][j], layers_size[i+1]);
		}
	}
}

void trainMiniBatch(int start_idx, int end_idx)
{
	updateMiniBatch(start_idx, end_idx, 1);
}

int testAccuracy(int idx, int dataset_type, int threadId)
{
	forwardPass(idx, dataset_type, threadId);
	switch(dataset_type)
	{
		case 1: if(equals(a[threadId][num_layers - 2], y_train[idx], layers_size[num_layers - 1]))
					return 1;
				break;
		case 2: if(equals(a[threadId][num_layers - 2], y_val[idx], layers_size[num_layers - 1]))
					return 1;
				break;
		case 3: if(equals(a[threadId][num_layers - 2], y_test[idx], layers_size[num_layers - 1]))
					return 1;
				break;
	}
	
	return 0;
}

float testEntr(int idx, int dataset_type)
{
	forwardPass(idx, dataset_type, 0);
	switch(dataset_type)
	{
		case 1:	
				return costFnLMS(y_train[idx], a[0][num_layers - 2], layers_size[num_layers - 1]);
		case 2: 
				return costFnLMS(y_val[idx], a[0][num_layers - 2], layers_size[num_layers - 1]);
		case 3: 
				return costFnLMS(y_test[idx], a[0][num_layers - 2], layers_size[num_layers - 1]);
	}
	
	return 0;
}

void *testAccuracyPerThread(void *arg)
{
	int procNo = (*(thread_params*) arg).procNo;
	int startId = (*(thread_params*) arg).startId;
	int endId = (*(thread_params*) arg).endId;
	int dataset_type = (*(thread_params*) arg).dataset_type;
	accur[procNo] = 0;
	for(int i = startId; i <= endId; i++)
		accur[procNo] += testAccuracy(i, dataset_type, procNo);
	accur[procNo] /= (endId - startId + 1);	 
}

float testBatchAccuracy(int start_idx, int end_idx, int dataset_type)
{
	pthread_t thd[NUM_THREADS];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	int ex_per_proc = (end_idx - start_idx + 1)/NUM_THREADS;
	for(unsigned procNo = 0; procNo < NUM_THREADS; procNo++)
	{	
		thread_params *curr_thread_params = (thread_params *)malloc(sizeof(thread_params));
		(*curr_thread_params).procNo = procNo;
		(*curr_thread_params).startId = start_idx + procNo*ex_per_proc;
		if(procNo != NUM_THREADS - 1)
			(*curr_thread_params).endId = start_idx + (procNo + 1)*ex_per_proc - 1;
		else
			(*curr_thread_params).endId = end_idx;
		(*curr_thread_params).dataset_type = dataset_type;
		pthread_create(&thd[procNo], &attr, testAccuracyPerThread, (void *)curr_thread_params);
	}
	for(unsigned procNo = 0; procNo < NUM_THREADS; procNo++)
	{
		pthread_join(thd[procNo],NULL);
	}
	
	float accuracy = 0;
	for(int t = 0; t <= NUM_THREADS; t++)
		accuracy += accur[t];
	accuracy /= NUM_THREADS;
	return accuracy;
}

float testBatchEntr(int start_idx, int end_idx, int dataset_type)
{
	float entr = 0;
	for(int i = start_idx; i <= end_idx; i++)
		entr += testEntr(i, dataset_type);
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
//	ofstream fout("entr_train.log");
	double timePerEpoch = 0.0;
	for(int epoch = 0; epoch < nEpochs; epoch++)
	{
		Timer.reset();
		int i;
		for(i = 0; i < numMiniBatches - 1; i++)
			trainMiniBatch(i*miniBatchSize, (i+1)*miniBatchSize - 1);
		trainMiniBatch(i*miniBatchSize, NUM_TRAIN - 1);
		cout  << "Epoch: " << epoch << endl;
		
		if(epoch > 0)
		{
			cout << "\t\t"; 
			timePerEpoch += Timer.lap("secs");
		}
		//entr = 	testBatchEntr(0, NUM_TRAIN - 1, 1);	
		
	//	cout << "\t\tTrain entr = " << entr << endl;		
	//	fout << entr << endl;
		accuracy = testBatchAccuracy(0, NUM_VAL - 1, 2);
		cout  << "\t\tValidation accuracy = " << accuracy*100 << "%" << endl;
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
}
