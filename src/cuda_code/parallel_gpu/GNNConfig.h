#ifndef GNNCONFIG_H
#define GNNCONFIG_H

/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * CUDA neural network definitions.
 */

#include<vector>

#include "../common/CudaHelper.h"
#include "../common/IOTools.h"

#define CUDA_DEVICE 0

enum UnitTest{
		MMUL,//MATRIX MULTIPLICATION
		TMMUL,//TRANSPOSE MATRIX MULTIPLICATION
		OUTDM,//OUTPUT DELTA COMPUTATION
		MHPROD,//MATRIX HADAMARD PRODUCT
		TVECPVEC// TRANSPOSE VECTOR PRODUCT VECTOR
	};

#define ZEROS 0
#define ONES 1
#define RANDOM 2
#define DEBUG_GNN false
#define DEBUG_T false

namespace gnn_actf{
	struct Sigmoid{
		char TAG[10] = "Sigmoid";
		template<typename T>
		T f(T x){
			return 1.0/(1.0 + exp(-x));
		}

		template<typename T>
		T d(T x){
			return f(x) * (1.0 - f(x));
		}

		template<typename T>
		__forceinline__ __device__ T D(T x){
			return F(x) * (1.0 - F(x));
		}

		template<typename T>
		__forceinline__ __device__ T F(T x){
			return 1.0/(1.0 + expf(-x));
		}
	};

	struct FSigmoid{
		char TAG[10] = "FSigmoid";

		template<typename T>
		T f(T x){
			return x/ (1.0 + fabs(x));
		}

		template<typename T>
		T d(T x){
			return 1.0 / pow(1.0 + fabs(x),2.0);
		}

		template<typename T>
		__forceinline__ __device__ T D(T x){
			return 1.0/powf(1.0 + fabsf(x),2.0);
		}

		template<typename T>
		__forceinline__ __device__ T F(T x){
			return x/(1.0 + fabsf(x));
		}
	};

	struct Arctan{
		char TAG[10] = "Arctan";

		template<typename T>
		T d(T x){
			return powf(1/acosh(x),2.0);
		}

		template<typename T>
		T f(T x){
			return 	tanh(x);
		}

		template<typename T>
		__forceinline__ __device__ T D(T x){
			return powf(1/acoshf(x),2.0);
		}

		template<typename T>
		__forceinline__ __device__ T F(T x){
			return 	tanhf(x);
		}
	};
}

namespace gnn_data{

	template<typename DATA_T>
	struct LayerBatch{
		unsigned int bsize;
		unsigned int clayer;

		DATA_T *A_j;
		DATA_T *D_j;
		DATA_T *Y;
		/*
		 * Description:
		 * 		Matrix of activation vectors for layer j.
		 * 		Columns = number of train examples in the batch
		 * 		Rows = number of neurons in the current layer.
		 * Notes:
		 *		First layer matrix size = number of input neurons x batch size
		 */
		LayerBatch(){
		}

		unsigned int initLayerBatch(unsigned clz,unsigned int bz, bool input){
			bsize = bz;
			clayer = clz;
			allocDevMem<DATA_T>(&A_j,sizeof(DATA_T)*bsize*clayer,"Error Allocating Activation Layer Batch Matrix");
			if(!input) allocDevMem<DATA_T>(&D_j,sizeof(DATA_T)*bsize*clayer,"Error Allocating Delta Layer Batch Matrix");
			return sizeof(DATA_T)*bsize*clayer + input ? 0 : sizeof(DATA_T)*bsize*clayer;

		}

		unsigned int initOutputBatch(){
			allocDevMem<DATA_T>(&Y, sizeof(DATA_T)*clayer * bsize, "Error allocating Output Y matrix");
			return sizeof(DATA_T)*clayer * bsize;
		}

		~LayerBatch(){
			cudaFree(A_j);
		}
	};

	template<typename DATA_T>
		struct Layer{
			unsigned int clayer;
			unsigned int nlayer;
			DATA_T *W_j = NULL;

			/*
			 * Description:
			 * 		Neural network layer with input weight matrix.
			 * 		Size of matrix is input neurons x output_neurons.
			 */
			Layer(){

			}

			unsigned int initLayer(unsigned int clz, unsigned int nlz){
				clayer = clz;
				nlayer = nlz;
				allocDevMem<DATA_T>(&W_j, sizeof(DATA_T)*clayer*nlayer, "Error Allocating Weight Matrix");
				return sizeof(DATA_T)*clayer*nlayer;
			}

			~Layer(){
				if(W_j != NULL)cudaFree(W_j);
			}

		};
}

namespace gnn{
	template<typename DATA_T, typename ACT_F>
	class GNeuralNetwork{
		public:
			GNeuralNetwork(ACT_F F){
				this->F = F;
				cudaSetDevice(CUDA_DEVICE);
			};

			~GNeuralNetwork(){
				if(network != NULL) delete[] network;
				if(hExamples != NULL) cudaFreeHost(hExamples);
				if(batch != NULL) delete[] batch;
				if(hTest !=NULL) cudaFreeHost(hTest);
			}

			void createLayers(std::vector<int> layers);
			void loadExamplesFromFile(std::string file);
			void loadTestExamplesFromFile(std::string file);
			void train();

			void setBatchSize(unsigned int bz){ this->bsize = bz; }
			void useTranspose(bool transpose){ this->transpose = transpose; }
			void setLearningRate(float lrate){ this->lrate = lrate; }

			void printConfig(double tt){
				unsigned int weights = 0;
				unsigned long rflops=0;
				for(int i = 0;i < layers-1 ;i++){
					weights+= network[i].nlayer * network[i].clayer;
					rflops+= network[i].nlayer * network[i].clayer * 2;
				}
				rflops*=  dimEx.first * 5;
				std::cout<< "Layers: " << layers-1 << std::endl;
				std::cout<< "Total weight number: " << weights << std::endl;
				std::cout << "Batch size: " << bsize <<std::endl;
				std::cout << "Elapsed time per training iteration (ms): " << tt <<std::endl;
				std::cout << "Required FLOP per iteration: " << rflops<< std::endl;
				double GFLOPS = ((double)rflops/(tt/1000))/1000000000;
				printf( "Achieved GFLOPS: %.f\n",GFLOPS);
			}

			/*
			 * Testing methods
			 */
			void bench_act();
			void print_weights();
			void bench_test_kernels(UnitTest test,unsigned int m, unsigned int n, unsigned int k, bool debug);
			void classify();
			bool validateInput(){
				if(network == NULL) vz::error("Initialize neural network before validation\n");
				//std::cout << network[0].clayer-1 + network[layers-2].nlayer <<std::endl;
				return (network[0].clayer-1 + network[layers-2].nlayer == dimEx.second);
			};

		private:
			unsigned int createLayerBatch();
			void randomInit();

			unsigned int layers = 0;
			unsigned int bsize = 0;//default value.
			float lrate =0.314;
			bool transpose = true;
			unsigned int mem = 0;

			arr2D dimEx;
			arr2D dimT;
			gnn_data::LayerBatch<DATA_T> *batch = NULL;
			gnn_data::Layer<DATA_T> *network = NULL;
			DATA_T *hExamples = NULL, *dExamples = NULL;
			DATA_T *hTest = NULL, *dTest = NULL;
			ACT_F F;
	};

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::loadExamplesFromFile(std::string file){
		cudaSetDevice(CUDA_DEVICE);
		IOTools<DATA_T> iot;
		dimEx = iot.dataDim(file);
		std::cout << "Loading examples from: " << file << std::endl;
		//std::cout<<dimEx.first << "," << dimEx.second << std::endl;
		iot.freadFile(hExamples,file,true);
		allocDevMem<DATA_T>(&dExamples,sizeof(DATA_T)*dimEx.first*dimEx.second,"Error Allocating dExamples memory");
		safeCpyToDevice<DATA_T>(dExamples,hExamples,sizeof(DATA_T)*dimEx.first*dimEx.second,"Error copying data to dExamples");
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::loadTestExamplesFromFile(std::string file){
		cudaSetDevice(CUDA_DEVICE);
		IOTools<DATA_T> iot;
		dimT = iot.dataDim(file);
		std::cout << "Loading test data from: " << file << std::endl;
		//std::cout<<dimEx.first << "," << dimEx.second << std::endl;
		iot.freadFile(hTest,file,true);
		allocDevMem<DATA_T>(&dTest,sizeof(DATA_T)*dimT.first*dimT.second,"Error Allocating dTest memory");
		safeCpyToDevice<DATA_T>(dTest,hTest,sizeof(DATA_T)*dimT.first*dimT.second,"Error copying data to dTest");
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::createLayers(std::vector<int> layers){
		cudaSetDevice(CUDA_DEVICE);
		if(layers.size() <= 0 ) vz::error("Network architecture not valid!");
		this->layers = layers.size();
		network = new gnn_data::Layer<DATA_T>[this->layers-1];

		/*clayer+1 = Weight matrix includes additional column for bias. nlayer x (clayer + 1)*/
		for(int i = 0;i<this->layers-1;i++) mem+=network[i].initLayer(layers[i]+1,layers[i+1]);
		randomInit();
		createLayerBatch();
	}

	/*
	 * For every batch create multi-layer matrices. Each batch will result in a matrix of activation vectors for each
	 * layer.
	 */
	template<typename DATA_T, typename ACT_F>
	unsigned int GNeuralNetwork<DATA_T,ACT_F>::createLayerBatch(){
		cudaSetDevice(CUDA_DEVICE);
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		if(hExamples == NULL) vz::error("Examples not loaded. Use loadExamplesFromFile!");
		//if(bsize > dimEx.first) bsize = dimEx.first;
		if(batch != NULL) delete[] batch;
		batch = new gnn_data::LayerBatch<DATA_T>[this->layers];

		/*(clayer - 1) = Activation does not include bias vector*/
		//printf("batch_size: %d\n",this->bsize);
		mem+=batch[0].initLayerBatch(network[0].clayer-1,this->bsize,true);
		/*nlayer is current layer without bias vector for activation matrix*/
		for(int i = 0; i < this->layers-1;i++) mem+=batch[i+1].initLayerBatch(network[i].nlayer,this->bsize,false);
		mem+=batch[this->layers-1].initOutputBatch();//Initialize Y Matrix
		return mem;
	}
}


#endif
