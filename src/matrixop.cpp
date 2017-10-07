#include <cmath>
#include "matrixop.h"
#define EPSILON 1e-10
#define INF 1e10

void prod(float *x, float s, float *y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = x[i]*s;
}

void prod(float s, float *x, float *y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = x[i]*s;
}

void mvProd(float **M, float *x, float *y, int m, int n)
{
	float sum = 0;
	for(int i = 0; i < m; i++)
	{
		sum = 0;		
		for(int j = 0; j < n; j++)
			sum += M[i][j] * x[j];			
		y[i] = sum;
	}
}

void mvProdT(float **M, float *x, float *y, int m, int n)
{
	int T = 500;
	float sum = 0;
	for(int j = 0; j < n; j++)
		y[j] = 0;
	for(int i = 0; i < m; i += T)
	{
		int i_max = (i + T > m) ? m : (i + T);
		for(int j = 0; j < n; j++)
		{
			sum = 0;
			for(int ii = i; ii < i_max; ii++)
				sum += M[ii][j] * x[ii];
			y[j] += sum;
		}		
	}
}

void add(float *x1, float *x2, float* y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = x1[i] + x2[i];
}

void hprod(float *x1, float *x2, float *y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = x1[i]*x2[i];
}

void sigmoid(float *x, float *y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = 1/(1 + exp(-x[i]));
}

void dSigmoid(float *x, float *y, int n)
{
	float temp = 0;
	for(int i = 0; i < n; i++)
	{
		temp = 1/(1 + exp(-x[i]));
		y[i] = temp * (1 - temp);
	}
}

void fastSigmoid(float *x, float *y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = x[i]/(1 + std::abs(x[i]));;
}

void dfastSigmoid(float *x, float *y, int n)
{
	float temp = 0;
	for(int i = 0; i < n; i++)
		y[i] = 1/pow(1 + std::abs(x[i]),2);
}

void softmax(float *x, float *y, int n)
{
	float sum = 0;
	for(int i = 0; i < n; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for(int i = 0; i < n; i++)
		y[i] /= sum;
}

void softmaxD(float *x, float *y, int n)
{
	float sum = 0;
	for(int i = 0; i < n; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for(int i = 0; i < n; i++)
	{
		y[i] /= sum;
		y[i] = y[i] * (1 - y[i]);
	}
}

float costFnXent(float *y_train, float *y_pred, int n_out)
{
	float cost = 0;
	for(int i = 0; i < n_out; i++)
		cost -= (y_train[i] * log2(y_pred[i]));
	
	return cost;
}

void costFnXentD(float *y_train, float *y_pred, float *delC_a, int n)
{
	for(int i = 0; i < n; i++)		
		delC_a[i] = (y_pred[i] > EPSILON) ? -y_train[i]/y_pred[i] : ((y_train[i] < EPSILON) ? 0 : -INF);
}

float costFnLMS(float *y_train, float *y_pred, int n_out)
{
	float cost = 0;
	for(int i = 0; i < n_out; i++)
		cost += (y_train[i] - y_pred[i])*(y_train[i] - y_pred[i]);
	
	return cost/2;
}

void costFnLMSD(float *y_train, float *y_pred, float *delC_a, int n)
{
	for(int i = 0; i < n; i++)		
		delC_a[i] = (y_pred[i] - y_train[i]);
}

bool equals(float* pred, float* label, int n)
{
	float max = -1;
	int maxpos = -1;
	for(int i = 0; i < n; i++)
	{
		if(pred[i] > max)
		{
			max = pred[i];
			maxpos = i;
		}
	}
	return (label[maxpos] == 1);
}