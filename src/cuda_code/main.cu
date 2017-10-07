/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * Testing and benchmarking neural network training.
 */

#include"common/ArgParser.h"
#include"common/Utils.h"
#include"parallel_gpu/GNNConfig.h"
#include<iostream>

void benchmarkKernels(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	s.bench_test_kernels(MMUL,2048,2048,2048,false);
	s.bench_test_kernels(TMMUL,2048,2048,2048,false);
	s.bench_test_kernels(MHPROD,2048,2048,2048, false);
	s.bench_test_kernels(TVECPVEC,2048,2048,2048,false);
}

void nn1(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	if(!ap.exists(F1ARG)) vz::error("Please use -f1= to provide a .csv file with the training examples");
	if(!ap.exists(F2ARG)) vz::error("Please use -f2= to provide a .csv file with the test examples");
	if(!ap.exists(INARG)) vz::error("Please specify input neuron number using -in=");
	if(!ap.exists(ONARG)) vz::error("Please specify output neuron number using -on=");

	std::string f1 = ap.getString(F1ARG);
	std::string f2 = ap.getString(F2ARG);
	unsigned int inn = ap.getUint(INARG);
	unsigned int onn = ap.getUint(ONARG);

	Time<millis> t;
	t.start();
	s.loadExamplesFromFile(f1);
	s.loadTestExamplesFromFile(f2);
	t.lap("Read Train and Test Data");

	std::vector<int> layers;
	layers.push_back(inn);//Input Layer//784
	layers.push_back(1024);//Hidden Layer
	layers.push_back(onn);//Output Layer//10

	unsigned int iterations = ap.exists(IARG) ? ap.getUint(IARG) : 50 ;
	unsigned int b = ap.exists(BARG) ? ap.getUint(BARG) : 100 ;
	float r = ap.exists(DARG) ? ap.getFloat(DARG) : 0.1 ;

	s.setBatchSize(b);
	s.useTranspose(true);
	s.setLearningRate(r);
	s.createLayers(layers);
	if(!s.validateInput()) vz::error("Input + Ouput Neurons != number of features");

	std::cout<<"Training...";
	t.start();
	for(int i = 0;i<iterations;i++){ s.train(); } std::cout << std::endl;
	s.printConfig(t.lap("Training Execution Time(ms)")/iterations);

	t.start();
	std::cout<<"Computing Classification Accuracy..." << std::endl;
	s.classify();
	t.lap("Classification Elapsed Time (ms)");
}

void nn2(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);


	if(!ap.exists(F1ARG)) vz::error("Please use -f1= to provide a .csv file with the training examples");
	if(!ap.exists(F2ARG)) vz::error("Please use -f2= to provide a .csv file with the test examples");
	if(!ap.exists(INARG)) vz::error("Please specify input neuron number using -in=");
	if(!ap.exists(ONARG)) vz::error("Please specify output neuron number using -on=");

	std::string f1 = ap.getString(F1ARG);
	std::string f2 = ap.getString(F2ARG);
	unsigned int inn = ap.getUint(INARG);
	unsigned int onn = ap.getUint(ONARG);

	Time<millis> t;
	t.start();
	s.loadExamplesFromFile(f1);
	s.loadTestExamplesFromFile(f2);
	t.lap("Read Train and Test Data");

	std::vector<int> layers;
	layers.push_back(inn);//Input Layer
	layers.push_back(1024);//Hidden Layer 1
	layers.push_back(1024);//Hidden Layer 2
	layers.push_back(onn);//Output Layer

	unsigned int iterations = ap.exists(IARG) ? ap.getUint(IARG) : 50 ;
	unsigned int b = ap.exists(BARG) ? ap.getUint(BARG) : 100 ;
	float r = ap.exists(DARG) ? ap.getFloat(DARG) : 0.1 ;

	s.setBatchSize(b);
	s.useTranspose(true);
	s.setLearningRate(r);
	s.createLayers(layers);
	if(!s.validateInput()) vz::error("Input + Ouput Neurons != number of features");

	std::cout<<"Training...";
	t.start();
	for(int i = 0;i<iterations;i++){ s.train(); } std::cout << std::endl;
	s.printConfig(t.lap("Training Execution Time(ms)")/iterations);

	t.start();
	std::cout<<"Computing Classification Accuracy..." << std::endl;
	s.classify();
	t.lap("Classification Elapsed Time");
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(ap.count()==0){ ap.menu();  return 0;}

	int mode = ap.exists(MDARG) ? ap.getUint(MDARG) : 0 ;
	if(mode == 0)	benchmarkKernels(ap);
	else if(mode==1) nn1(ap);
	else if(mode==2) nn2(ap);
}
