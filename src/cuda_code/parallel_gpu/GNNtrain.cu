#include "GNNConfig.h"
#include "../common/Time.h"

/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * CUDA neural network implementation
 */

#define MBATCH 128
#define TTILE 32
#define LTILE 32

#define DPT 4 //DATA PER THREADS
#define BSIZE 512

namespace gnn_kernels{
	static __device__ curandState randDevStates[RAND_STATES];

	__device__ float cudaUniRand(unsigned int tid){
		return curand_uniform(&randDevStates[tid % RAND_STATES]);
	}

	__global__ void cudaSetupRandStatesKernel(unsigned int seed){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, blockIdx.x, 0, &randDevStates[i]);
	}

	__host__ void cudaInitRandStates(){
		dim3 grid = grid_1D(RAND_STATES,RAND_BLOCK_THREADS);
		dim3 block = block_1D(RAND_BLOCK_THREADS);

		Utils<unsigned int> u;
		cudaSetupRandStatesKernel<<<grid,block>>>(u.uni(UINT_MAX));
		handleDeviceErrors(cudaDeviceSynchronize(),"Error initializing random states");
	}


	/*
	 * Testing activation functions on kernels.
	 */
	template<typename ACT_F>
	__global__ void bench_test_activation(ACT_F F){
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		float a = 0;
		for(int j = 0; j<blockDim.x;j++){
			a+= F.F((float)i*j);
		}
	}

	/*
	 * Initialize matrices random weights
	 */
	template<typename DATA_T>
	__global__ void randomWeights(DATA_T *W_j,unsigned int rows, unsigned int cols){
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if( i < rows * cols){
			if((i+1) % cols == 0){
				W_j[i] = 0.0;
			}else{
				W_j[i] = (2.0 * cudaUniRand(i) - 1.0) * (sqrtf(6.0)/ (rows + cols));
			}
		}
	}
	/*
	 * Load current batch of train examples.
	 * 		1: First layer batch array.
	 * 		2: Training example matrix.
	 * 		3: Input layer dimension
	 * 		4: Batch size dimension
	 * 		5: Offset indicating the batch being loaded.
	 * 	Notes:
	 * 		Transpose version assumes that the training examples matrix is stored
	 * 		in a row-wise manner.
	 */
	template<typename DATA_T, unsigned int TILE>
	__global__ void loadT(
			DATA_T *A_j,
			DATA_T *tEx,
			unsigned int clayer, unsigned int bsize,
			unsigned int car, unsigned int dim,
			unsigned int voffset, unsigned int hoffset
			){

		__shared__ DATA_T stEx[TILE * TILE];
		int row = ( blockIdx.y * blockDim.y + threadIdx.y );
		int col = ( blockIdx.x * blockDim.x + threadIdx.x );

		if(voffset + row < car && col + hoffset < dim && row < bsize && col < clayer){
			stEx[threadIdx.y * TILE + threadIdx.x] = tEx[(row + voffset) * dim + (col + hoffset)];
			//stEx[threadIdx.y * TILE + threadIdx.x] = tEx[(voffset + row) * dim + (col + hoffset)];
		}
		__syncthreads();

		//col * bsize + row
		row = (blockIdx.x * blockDim.x + threadIdx.y);
		col = (blockIdx.y * blockDim.y + threadIdx.x);
		if( row < clayer && col < bsize){
			A_j[row * bsize + col] = stEx[threadIdx.x * TILE + threadIdx.y];
		}
	}

	/*
	 * Compute matrix of activation values for a single layer of a given batch.
	 *		1:	Current layer weight matrix.
	 *		2: 	Current layer matrix of activation vectors.
	 *		3: 	Next layer matrix of activation vectors.
	 *		4:	W_j = nlayer x clayer , A_j = clayer x bsize, A_jj = nlayer x bsize.
	 *		5: 	Offset: 0 for hidden and output layer, corresponding row of training example matrix for input layer.
	 */
	template<typename DATA_T, typename ACT_F, unsigned int TILE>
	__global__ void	mmul(
			DATA_T *A_jj,
			DATA_T *W_j,
			DATA_T *A_j,
			ACT_F F,
			unsigned int nlayer,
			unsigned int clayer,
			unsigned int bsize
			)
	{
		__shared__ DATA_T sWj[TILE * TILE];
		__shared__ DATA_T sAj[TILE * TILE];
		__shared__ DATA_T bias[TILE];

		int row = ( blockIdx.y * blockDim.y + threadIdx.y );
		int col = ( blockIdx.x * blockDim.x + threadIdx.x );
		if(threadIdx.x == 0) bias[threadIdx.y] = W_j[row * (clayer + 1) + clayer];
		__syncthreads();
		DATA_T Ajj = bias[threadIdx.y];

		int loadOffset = threadIdx.y*TILE + threadIdx.x;
		for(int i = 0;i < ((clayer - 1) / TILE) + 1; i++){
			if( row < nlayer && (i * TILE + threadIdx.x ) < clayer)
				sWj[loadOffset] = W_j[ row * ( clayer + 1 ) + i * TILE  + threadIdx.x];// clayer + 1  to avoid bias vector
			else sWj[loadOffset] = 0.0;

			if ( i*TILE + threadIdx.y < clayer && col < bsize )
				sAj[loadOffset] = A_j[(i * TILE + threadIdx.y) * bsize + col];
			else sAj[loadOffset] = 0.0;
			__syncthreads();

			for(int j = 0;j < TILE; j++){
				Ajj += sWj[threadIdx.y * TILE + j] * sAj[j * TILE + threadIdx.x];
			}

			__syncthreads();
		}

		if( row < nlayer && col < bsize )
			A_jj[row * bsize + col ] = F.F(Ajj);
	}

	/*
	 * Kernel that computes the last layer difference between the batch activation matrix and the expected output
	 * matrix.
	 */
	template<typename DATA_T>
	__global__ void outputD(
			DATA_T *D_j,
			DATA_T *ExA_j,
			DATA_T *A_j,
			unsigned int size
		)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < size){ D_j[i] = ExA_j[i] - A_j[i]; }
	}

	/*
	 * Transpose matrix multiplication.
	 *  D_j = (W_j)^T . D_jj
	 */
	template<typename DATA_T, unsigned int TILE>
	__global__ void tmmul(
			DATA_T *D_j,
			DATA_T *W_j,
			DATA_T *D_jj,
			unsigned int clayer,
			unsigned int nlayer,
			unsigned int bsize
			)
	{
		__shared__ DATA_T sWj[TILE * TILE];
		__shared__ DATA_T sDjj[TILE * TILE];

		DATA_T Dj = 0.0;
		int colW = ( blockIdx.y * blockDim.y + threadIdx.x );// by * TILE + ty * clayer + threadIdx.x
		int colD = ( blockIdx.x * blockDim.x + threadIdx.x );

		int loadOffset = threadIdx.y*TILE + threadIdx.x;
		for(int i = 0; i < (nlayer - 1) / TILE + 1 ; i++){
			if( (i * TILE +  threadIdx.y) < nlayer && colW < clayer)
				sWj[loadOffset] = W_j[ (i * TILE +  threadIdx.y) * clayer + colW ];
			else
				sWj[loadOffset] = 0.0;

			if((i * TILE + threadIdx.y) < nlayer && colD < bsize)
				sDjj[loadOffset] = D_jj[ (i * TILE + threadIdx.y) * bsize + colD ];
			else
				sDjj[loadOffset] = 0.0;
			__syncthreads();

			for(int j=0;j<TILE;j++) Dj += sWj[j * TILE + threadIdx.y] * sDjj[j * TILE + threadIdx.x];
			__syncthreads();
		}

		int row = ( blockIdx.y * blockDim.y + threadIdx.y );
		if( row < clayer && colD < bsize) D_j[row * bsize + colD] = Dj;
	}

	/*
	 *	Hadamard product with derivative activation values
	 */
	template<typename DATA_T, typename ACT_F, unsigned int TILE>
	__global__ void	hmprod_mmul(
			DATA_T *D_j,
			DATA_T *W_j,
			DATA_T *A_j,
			ACT_F F,
			unsigned int nlayer,
			unsigned int clayer,
			unsigned int bsize
			)
	{
		__shared__ DATA_T sWj[TILE * TILE];
		__shared__ DATA_T sAj[TILE * TILE];
		__shared__ DATA_T bias[TILE];

		int row = ( blockIdx.y * blockDim.y + threadIdx.y );
		int col = ( blockIdx.x * blockDim.x + threadIdx.x );
		if(threadIdx.x == 0) bias[threadIdx.y] = W_j[row * (clayer + 1) + clayer];
		__syncthreads();
		DATA_T Dj = bias[threadIdx.y];

		int loadOffset = threadIdx.y*TILE + threadIdx.x;
		for(int i = 0;i < ((clayer - 1) / TILE) + 1; i++){
			if( row < nlayer && (i * TILE + threadIdx.x ) < clayer)
				sWj[loadOffset] = W_j[ row * ( clayer + 1 ) + i * TILE  + threadIdx.x];// clayer + 1  to avoid bias vector
			else sWj[loadOffset] = 0.0;

			if ( i*TILE + threadIdx.y < clayer && col < bsize )
				sAj[loadOffset] = A_j[(i * TILE + threadIdx.y) * bsize + col];
			else sAj[loadOffset] = 0.0;
			__syncthreads();

			for(int j = 0;j < TILE; j++) Dj += sWj[threadIdx.y * TILE + j] * sAj[j * TILE + threadIdx.x];
			__syncthreads();
		}


		if( row < nlayer && col < bsize )
			D_j[row * bsize + col ] *= F.D(Dj);
	}

	template<typename DATA_T, typename ACT_F, unsigned int TILE>
	__global__ void	hmprod_mmul_sigmoid(
			DATA_T *D_j,
			DATA_T *A_j,
			unsigned int nlayer,
			unsigned int clayer,
			unsigned int bsize
			)
	{

		int row = ( blockIdx.y * blockDim.y + threadIdx.y );
		int col = ( blockIdx.x * blockDim.x + threadIdx.x );
		DATA_T Aj = A_j[row * bsize + col];

		if( row < nlayer && col < bsize )
			D_j[row * bsize + col ] *= Aj * (1-Aj);
	}


	/*
	 * 	Compute weight update matrices for the current batch.
	 * 	A = [ A ones(bsize) ]
	 *	for i = 1 : dsz(2)
	 *		W = W + D(:,i) * A(:,i)';
	 *	end
	 *	W ( nlayer x (clayer + 1))
	 */
	template<typename DATA_T, unsigned int TILE>
	__global__ void tvecpvec(
			DATA_T *W_j,
			DATA_T *D_jj,
			DATA_T *A_j,
			unsigned int nlayer,
			unsigned int bsize,
			unsigned int clayer,
			float lrate
			){

		__shared__ DATA_T sDjj[TILE * TILE];
		__shared__ DATA_T sAj[TILE * TILE];

		DATA_T Wj = 0.0;
		int rowD = (blockIdx.y * blockDim.y + threadIdx.y);
		int rowA = (blockIdx.x * blockDim.x + threadIdx.y);

		for(int i = 0;i < (bsize - 1) / TILE + 1;i++){
			if(rowD < nlayer && (i*TILE + threadIdx.x) < bsize)
				sDjj[threadIdx.y * TILE + threadIdx.x] = D_jj[rowD * bsize + i*TILE + threadIdx.x];
			else
				sDjj[threadIdx.y * TILE + threadIdx.x] = 0.0;

			if(rowA < clayer && (i*TILE + threadIdx.x) < bsize)
				sAj[threadIdx.x * TILE + threadIdx.y] = A_j[rowA * bsize + i*TILE + threadIdx.x];
			else
				sAj[threadIdx.x * TILE + threadIdx.y] = 1.0;//Required to update bias weights//
			__syncthreads();

			for(int j = 0 ; j < TILE; j++)
				Wj += sDjj[threadIdx.y * TILE + j] * sAj[j * TILE + threadIdx.x];
			__syncthreads();
		}

		int col = (blockIdx.x * blockDim.x + threadIdx.x);
		Wj *= (lrate / bsize);
		if( rowD < nlayer && col < clayer + 1)//clayer + 1 to update bias weights.
			W_j[rowD * (clayer + 1) + col] += Wj;
	}

	template<typename DATA_T,unsigned int init>
	__global__ void initVector(DATA_T *M, unsigned int rows, unsigned int cols){
		int i = threadIdx.x + blockDim.x * blockIdx.x;

		while( i < rows * cols){
			if (init == ZEROS ) M[i] = 0.0;
			else if (init == ONES) M[i] = 1.0;
			else if (init == RANDOM) M[i] = cudaUniRand(i);
			i+=gridDim.x * blockDim.x;
		}
	}


	template<typename DATA_T>
	__global__ void printGPU2(DATA_T *A, unsigned int row, unsigned int col){
		for(int i =0;i<row;i++){
			printf("[ ");
			for(int j=0;j<col;j++){
				printf("%.1f ", A[i*col + j]);
			}
			printf(" ]\n");
		}
	}

	template<typename DATA_T>
	__host__ void printGPU(DATA_T *A, unsigned int row, unsigned int col){
		DATA_T *AA;
		allocHostMem<DATA_T>(&AA,sizeof(DATA_T)*row*col,"Error allocating AA in printGPU");
		safeCpyToHost<DATA_T>(AA,A,sizeof(DATA_T)*row*col,"Error copying AA in printGPU");
		printf("[ ");
		for(int i =0;i<row;i++){
			for(int j=0;j<col;j++){
				printf("%.1f ", AA[i*col + j]);
			}
			printf(" ;\n");
		}
		printf(" ]\n");
	}
}

namespace gnn{

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::randomInit(){
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		//std::cout<<"Initializing random weights: "<<std::endl;
		cudaSetDevice(CUDA_DEVICE);

		gnn_kernels::cudaInitRandStates();
		for(int i = 0;i < layers-1;i++){
			//std::cout<<network[i].clayer << "{}" << network[i].nlayer << std::endl;
			unsigned int vector_size = network[i].nlayer * network[i].clayer;
			dim3 grid = grid_1D(vector_size,BSIZE);
			dim3 block = block_1D(BSIZE);
			gnn_kernels::randomWeights<DATA_T><<<grid,block>>>(network[i].W_j,network[i].nlayer,network[i].clayer);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing randomWeights kernel");
		}
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::train(){
		cudaSetDevice(CUDA_DEVICE);
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		if(bsize == 0) vz::error("Batch size not set. Use setBatchSize first!");
		unsigned int nbatch = this->transpose ? dimEx.first / this->bsize : dimEx.second / this->bsize;

		for(int i = 0; i< nbatch; i++){
			/*
			 * Load current batch of training examples.
			 */
			unsigned int bRow = i * this->bsize;
			dim3 lgrid((batch[0].clayer-1)/LTILE + 1, (batch[0].bsize-1)/LTILE + 1);
			dim3 lblock(LTILE,LTILE);
			gnn_kernels::loadT<DATA_T,LTILE><<<lgrid,lblock>>>(
					batch[0].A_j,dExamples,
					batch[0].clayer,batch[0].bsize,
					dimEx.first,dimEx.second,
					bRow,0
				);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing loadT for X batch");

			if(DEBUG_T){
			//printf("A=");
			gnn_kernels::printGPU(batch[0].A_j,batch[0].clayer,batch[0].bsize);
			cudaDeviceSynchronize();
			printf("sum(sum(round(A - M(%d:%d,1:%d)')))\n",bRow + 1, (i+1)*batch[0].bsize,batch[0].clayer);
			}


			/*
			 * Neural network feed forward step.
			 * 		- W = ( nlayer x (clayer + 1) ), A(i) = ( clayer x bsize ) , A(i+1) = (nlayer x bsize)
			 * 		A[jj] = A[j] * W[j]
			 */
			for(int j = 0;j < this->layers - 1;j++){
				dim3 agrid((batch[j+1].bsize - 1)/TTILE + 1, (batch[j+1].clayer - 1)/TTILE + 1);
				dim3 ablock(TTILE,TTILE);
				gnn_kernels::mmul<DATA_T,ACT_F,TTILE><<<agrid,ablock>>>
						(
								batch[j+1].A_j,
								network[j].W_j,
								batch[j].A_j,
								F,
								network[j].nlayer,
								network[j].clayer - 1,// Ignore bias vector from the multiplication//
								batch[j].bsize
						);
				handleDeviceErrors(cudaDeviceSynchronize(),"Error executing batch activation");

				if(DEBUG_GNN){
				printf("Ajj= ");
				gnn_kernels::printGPU(batch[j+1].A_j,batch[j+1].clayer,batch[j+1].bsize);
				cudaDeviceSynchronize(); //printf("------------------>\n");
				printf(";W= ");
				gnn_kernels::printGPU(network[j].W_j,network[j].nlayer,network[j].clayer);
				cudaDeviceSynchronize(); //printf("------------------>\n");
				printf(";Aj= ");
				gnn_kernels::printGPU(batch[j].A_j,batch[j].clayer,batch[j].bsize);
				cudaDeviceSynchronize();
				printf("Ejj=act(W,Aj,0);\n");
				printf("diff0%d = sum(sum(round(Ejj-Ajj)))\n",j);
				}
			}

			/*
			 * Output layer Delta computation.
			 *	Dl = (Y - Al)
			 *
			 */
			dim3 ogrid = grid_1D(batch[layers-1].clayer * batch[layers-1].bsize, BSIZE);
			dim3 oblock = block_1D(BSIZE);

			bRow = i * this->bsize;
			dim3 lygrid((batch[layers-1].clayer-1)/LTILE + 1, (batch[layers-1].bsize-1)/LTILE + 1);
			dim3 lyblock(LTILE,LTILE);
			gnn_kernels::loadT<DATA_T,LTILE><<<lygrid,lyblock>>>(
					batch[layers-1].Y,dExamples,
					batch[layers-1].clayer,batch[layers-1].bsize,
					dimEx.first,dimEx.second,
					bRow,batch[0].clayer
			);

			if(DEBUG_T){
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing loadT for Y batch");
			printf("Y=");
			gnn_kernels::printGPU(batch[layers-1].Y,batch[layers-1].clayer,batch[layers-1].bsize);
			cudaDeviceSynchronize();
			printf("sum(sum(round(Y - M(%d:%d,%d:%d)')))\n",bRow + 1, (i+1)*batch[0].bsize,batch[0].clayer+1,batch[0].clayer+batch[layers-1].clayer);
			}

			gnn_kernels::outputD<DATA_T><<<ogrid,oblock>>>(
					batch[layers-1].D_j,
					batch[layers-1].Y,
					batch[layers-1].A_j,// Dj =  Y - Aj
					batch[layers-1].clayer * batch[layers-1].bsize
				);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing outputD kernel");

			if(DEBUG_GNN){
			printf("Y=");
			gnn_kernels::printGPU(batch[layers-1].Y,batch[layers-1].clayer,batch[layers-1].bsize);
			cudaDeviceSynchronize();
			//if(DEBUG_GNN){
			printf(";Aj=");
			gnn_kernels::printGPU(batch[layers-1].A_j,batch[layers-1].clayer,batch[layers-1].bsize);
			cudaDeviceSynchronize();
			printf(";Dl=");
			gnn_kernels::printGPU(batch[layers-1].D_j,batch[layers-1].clayer,batch[layers-1].bsize);
			cudaDeviceSynchronize();
			printf(";El = (Y - Aj);\n");
			printf("diffY = sum(sum(round(El - Dl)))\n");
			}

			/*
			 * Backpropagation transpose matrix multiplication.
			 * 		for i = layers-1 : > 1 : i--
			 * 		batch[i-1].D_j = network[i-1].W_j * batch[i].D_j
			 * 		grid = (batch[i-1].bsize / TILE + 1), batch[i-1].clayer / TILE + 1
			 * 		block = (TILE, TILE)
			 * 		D[j] = <W[j] * D[jj]> .* F.D(W[j] * A[j])
			 */
			for(int j = layers-1; j > 1 ; j--){
					//printf("BP(%d)\n",j);
					dim3 dgrid((batch[j-1].bsize - 1) / TTILE + 1, (batch[j-1].clayer - 1) / TTILE + 1);
					dim3 dblock(TTILE, TTILE);
					gnn_kernels::tmmul<DATA_T,TTILE><<<dgrid,dblock>>>(
							batch[j-1].D_j,//(clayer x bsize)
							network[j-1].W_j,//(nlayer x clayer)
							batch[j].D_j,// (nlayer x bsize)
							network[j-1].clayer,
							network[j-1].nlayer,
							batch[j].bsize
							);
					handleDeviceErrors(cudaDeviceSynchronize(),"Error executing tmmul kernel");
					if(DEBUG_GNN){
					printf("Djj=");
					gnn_kernels::printGPU(batch[j-1].D_j,batch[j-1].clayer,batch[j-1].bsize);
					cudaDeviceSynchronize(); //printf("------------------>\n");
					printf(";W=");
					gnn_kernels::printGPU(network[j-1].W_j,network[j-1].nlayer,network[j-1].clayer);
					cudaDeviceSynchronize();//printf("------------------>\n");
					printf(";Dj=");
					gnn_kernels::printGPU(batch[j].D_j,batch[j].clayer,batch[j].bsize);
					cudaDeviceSynchronize();//printf("------------------>\n");
					printf("Ejj=W' * Dj; \n");
					printf("diff1%d = sum(sum(round(Ejj(1:%d,:) - Djj)))\n",j-1,network[j-1].clayer-1);
					}
			}

			/*
			 * Final step for delta computation.
			 * 		//D[jj] = D[jj] .* F.D(W[j] * A(j))
			 */
			for(int j = 1; j < layers-1; j++){
				dim3 dgrid((batch[j].bsize - 1) / TTILE + 1, (batch[j].clayer - 1) / TTILE + 1);
				dim3 dblock(TTILE, TTILE);

				if(DEBUG_GNN){
					printf("Djj=");
					gnn_kernels::printGPU(batch[j].D_j,batch[j].clayer,batch[j].bsize);
					cudaDeviceSynchronize(); //printf("------------------>\n");
					printf(";Wj=");
					gnn_kernels::printGPU(network[j-1].W_j,network[j-1].nlayer,network[j-1].clayer);
					cudaDeviceSynchronize();//printf("------------------>\n");
					printf(";Aj=");
					gnn_kernels::printGPU(batch[j-1].A_j,batch[j-1].clayer,batch[j-1].bsize);
					cudaDeviceSynchronize();//printf("------------------>\n");
				}
				if(strcmp(F.TAG,"Sigmoid")!=0){
					gnn_kernels::hmprod_mmul<DATA_T,ACT_F,TTILE><<<dgrid, dblock>>>(
							batch[j].D_j,
							network[j-1].W_j,
							batch[j-1].A_j,
							F,
							network[j-1].nlayer,
							network[j-1].clayer-1,
							batch[j-1].bsize
							);
				}else{
					gnn_kernels::hmprod_mmul_sigmoid<DATA_T,ACT_F,TTILE><<<dgrid, dblock>>>(
							batch[j].D_j,
							batch[j].A_j,
							network[j-1].nlayer,
							network[j-1].clayer-1,
							batch[j-1].bsize
							);
				}
				handleDeviceErrors(cudaDeviceSynchronize(),"Error executing tmmul kernel");//TODO not necessary

				if(DEBUG_GNN){
				printf(";Ejj=");
				gnn_kernels::printGPU(batch[j].D_j,batch[j].clayer,batch[j].bsize);
				cudaDeviceSynchronize(); //printf("------------------>\n");
				printf(";Djj = Djj .* act(Wj,Aj,1);\n");
				printf("diff2%d=sum(sum(round(Ejj - Djj)))\n",j);
				}
			}

			/*
			 * Weight and bias update
			 * W[j] = W[j] + (lrate/bsize) * Sum( D[jj] <> A[j] )
			 */
			for(int j = 0;j<layers-1; j++){
				dim3 grid((network[j].clayer - 1)/TTILE + 1, (network[j].nlayer - 1)/TTILE + 1 );
				dim3 block(TTILE,TTILE);
				if(DEBUG_GNN){
					printf("Wj=");
					gnn_kernels::printGPU(network[j].W_j,network[j].nlayer,network[j].clayer);
					cudaDeviceSynchronize();
					printf(";Djj=");
					gnn_kernels::printGPU(batch[j+1].D_j,batch[j+1].clayer,batch[j+1].bsize);
					cudaDeviceSynchronize();
					printf(";Aj=");
					gnn_kernels::printGPU(batch[j].A_j,batch[j].clayer,batch[j].bsize);
					cudaDeviceSynchronize();
				}

				gnn_kernels::tvecpvec<DATA_T,TTILE><<<grid,block>>>(
					network[j].W_j,
					batch[j+1].D_j,
					batch[j].A_j,
					network[j].nlayer,
					batch[j].bsize,
					network[j].clayer-1,
					this->lrate
					);
				handleDeviceErrors(cudaDeviceSynchronize(),"Error executing tvecpvec kernel");//TODO not needed to wait
				if(DEBUG_GNN){
					printf("Ej=");
					gnn_kernels::printGPU(network[j].W_j,network[j].nlayer,network[j].clayer);
					cudaDeviceSynchronize();
					printf("diff3%d=sum(sum(round(Ej-tvecpvec(Wj,Djj,Aj,%f,%d))))\n",j,this->lrate,batch[j].bsize);
				}
			}
		}

	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::classify(){
		cudaSetDevice(CUDA_DEVICE);
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		if(bsize == 0) vz::error("Batch size not set. Use setBatchSize first!");
		unsigned int nbatch = this->transpose ? dimT.first / this->bsize : dimT.second / this->bsize;
		unsigned int count =0;
		if(DEBUG_GNN) std::cout<<dimT.first << "," << dimT.second << std::endl;
		if(DEBUG_GNN) std::cout<< "nbatch: " << nbatch << std::endl;

		DATA_T *Y,*A;
		allocHostMem<DATA_T>(&Y,sizeof(DATA_T)*this->bsize*batch[layers-1].clayer,"Error allocating mem for Y in classify");
		allocHostMem<DATA_T>(&A,sizeof(DATA_T)*this->bsize*batch[layers-1].clayer,"Error allocating mem for A in classify");
		for(int i = 0; i< nbatch; i++){
			/*
			 * Load current batch of training examples.
			 */
			if(this->transpose){
				unsigned int bRow = i * this->bsize;
				dim3 lgrid((batch[0].clayer-1)/TTILE + 1, (batch[0].bsize-1)/TTILE + 1);
				dim3 lblock(TTILE,TTILE);
				//print_grid(lgrid,lblock);
				gnn_kernels::loadT<DATA_T,TTILE><<<lgrid,lblock>>>(
						batch[0].A_j,dTest,
						batch[0].clayer,batch[0].bsize,
						dimT.first,dimT.second,
						bRow,0);
				handleDeviceErrors(cudaDeviceSynchronize(),"Error executing loadT for X batch on classify");
			}

			/*
			 * Neural network feed forward step.
			 * 		- W = ( nlayer x (clayer + 1) ), A(i) = ( clayer x bsize ) , A(i+1) = (nlayer x bsize)
			 * 		A[jj] = A[j] * W[j]
			 */
			for(int j = 0;j < this->layers - 1;j++){
				dim3 agrid((batch[j+1].bsize - 1)/TTILE + 1, (batch[j+1].clayer - 1)/TTILE + 1);
				dim3 ablock(TTILE,TTILE);
				gnn_kernels::mmul<DATA_T,ACT_F,TTILE><<<agrid,ablock>>>
						(
							batch[j+1].A_j,
							network[j].W_j,
							batch[j].A_j,
							F,
							network[j].nlayer,
							network[j].clayer - 1,// Ignore bias vector from the multiplication//
							batch[j].bsize
						);
				handleDeviceErrors(cudaDeviceSynchronize(),"Error executing batch activation");
			}

			unsigned int bRow = i * this->bsize;
			dim3 lgrid((batch[layers-1].clayer-1)/TTILE + 1, (batch[layers-1].bsize-1)/TTILE + 1);
			dim3 lblock(TTILE,TTILE);
			gnn_kernels::loadT<DATA_T,TTILE><<<lgrid,lblock>>>(
					batch[layers-1].Y,dTest,
					batch[layers-1].clayer,batch[layers-1].bsize,
					dimT.first,dimT.second,
					bRow,batch[0].clayer
			);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing loadT for Y batch on classify");

			safeCpyToHost<DATA_T>(Y,batch[layers-1].Y,sizeof(DATA_T)*batch[layers-1].clayer*this->bsize,"Error transferring Y from GPU");
			safeCpyToHost<DATA_T>(A,batch[layers-1].A_j,sizeof(DATA_T)*batch[layers-1].clayer*this->bsize,"Error transferring A_j from GPU");
			for(int x=0; x < bsize; x++){
				DATA_T maxY = 0, maxA=0;
				int indexY = 0, indexA=0;
				for(int y = 0; y < batch[layers-1].clayer; y++){
					if(Y[y * bsize + x] > maxY){ maxY = Y[y * bsize + x]; indexY = y;}
					if(A[y * bsize + x] > maxA){ maxA = A[y * bsize + x]; indexA = y;}
				}
				if(indexY == indexA ) count++;
			}
		}
		printf("Accuracy: %2.f, %u, %lu\n",(((float)count)/dimT.first)*100,count, dimT.first);
		cudaFreeHost(Y); cudaFreeHost(A);
	}

	/*
	 * Testing methods
	 */
	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::bench_act(){
		cudaSetDevice(0);
		dim3 block(512,1,1);
		dim3 grid(128,1,1);

		/*
		 * Warm up device
		 */
		gnn_kernels::bench_test_activation<ACT_F><<<grid,block>>>(this->F);
		cudaDeviceSynchronize();
		/* <END> */

		std::string msg("Benchmark ");
		msg.append(F.TAG);
		Time<millis> t;
		t.start();
		gnn_kernels::bench_test_activation<ACT_F><<<grid,block>>>(F);
		cudaDeviceSynchronize();
		t.lap(msg);
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::print_weights(){
		DATA_T *cW_j;
		cudaSetDevice(0);

		for(int i = 0;i < layers-1;i++){
			unsigned int size = network[i].nlayer * network[i].clayer;
			allocHostMem<DATA_T>(&cW_j,sizeof(DATA_T)*size, "Error Allocating Host Weight Matrix");
			safeCpyToHost<DATA_T>(cW_j,network[i].W_j,sizeof(DATA_T)*size, "Error Allocating Copying Weight Matrix From Device");

			printf("W%d=[",i);
			for(int j = 0;j<size;j++){
				std::cout<<cW_j[j] << " ";
				if((j+1)%network[i].clayer == 0) std::cout<<std::endl;
			}
			printf("]");
			std::cout<<std::endl;
		}

		for(int i = 0;i < layers-1;i++) printf("A%d=act(W%d,A%d,0)\n",i+1,i,i);

	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::bench_test_kernels(UnitTest test,unsigned int m, unsigned int n, unsigned int k,
			bool debug){
		unsigned int nlayer = m, clayer = n + 1, bsize = k;

		DATA_T *hostA, *hostB, *hostC, *hostD;
		DATA_T *devA, *devB, *devC;

		allocDevMem<DATA_T>(&devA,sizeof(DATA_T) * nlayer * clayer, "Error allocating devA memory");
		allocDevMem<DATA_T>(&devB,sizeof(DATA_T) * clayer * bsize, "Error allocating devB memory");
		allocDevMem<DATA_T>(&devC,sizeof(DATA_T) * nlayer * bsize, "Error allocating devC memory");

		allocHostMem<DATA_T>(&hostA,sizeof(DATA_T) * nlayer * clayer, "Error allocating devA memory");
		allocHostMem<DATA_T>(&hostB,sizeof(DATA_T) * clayer * bsize, "Error allocating devB memory");
		allocHostMem<DATA_T>(&hostC,sizeof(DATA_T) * nlayer * bsize, "Error allocating devC memory");

		dim3 rgrid;
		dim3 rblock = block_1D(256);
		rgrid = grid_1D(nlayer * clayer,256); gnn_kernels::randomWeights<DATA_T><<<rgrid,rblock>>>(devA,nlayer, clayer);
		rgrid = grid_1D(clayer * bsize,256); gnn_kernels::randomWeights<DATA_T><<<rgrid,rblock>>>(devB,clayer, bsize);
		rgrid = grid_1D(nlayer * bsize,256); gnn_kernels::randomWeights<DATA_T><<<rgrid,rblock>>>(devC,nlayer,bsize);

		if(test == MMUL){
			dim3 agrid((bsize - 1)/TTILE + 1, (nlayer - 1)/TTILE + 1);
			dim3 ablock(TTILE,TTILE);
			Time<millis> t;
			t.start();
			gnn_kernels::mmul<DATA_T,ACT_F,TTILE><<<agrid,ablock>>>
					(
							devC,
							devA,
							devB,
							F,
							nlayer,
							clayer - 1,
							bsize
					);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing batch mmul");
			t.lap("GPU serial mmul elapsed time");

			allocHostMem<DATA_T>(&hostD,sizeof(DATA_T) * nlayer * bsize, "Error allocating devC memory");
			safeCpyToHost<DATA_T>(hostA,devA,sizeof(DATA_T)*nlayer*clayer,"Error copying devA to host");
			safeCpyToHost<DATA_T>(hostB,devB,sizeof(DATA_T)*clayer*bsize,"Error copying devB to host");
			safeCpyToHost<DATA_T>(hostC,devC,sizeof(DATA_T)*nlayer*bsize,"Error copying devC to host");

			/*t.start();
			for(int x = 0; x < nlayer; x++){//3
				for (int y = 0; y < bsize; y++){//3
					hostD[x * bsize + y] = hostA[x * (clayer) + clayer - 1];
					for (int z = 0; z < clayer - 1; z++){//2
						hostD[x * bsize + y] += hostA[x * (clayer) + z] * hostB[z * bsize + y];
					}
					hostD[x * bsize + y] = F.f(hostD[x * bsize + y]);
				}
			}
			if(!debug) t.lap("CPU serial mmul elapsed time");
			if(debug){
				gnn_kernels::printGPU(devC,m,k);
				cudaDeviceSynchronize(); printf("<----->\n");
				gnn_kernels::printGPU(devA,m,n);
				cudaDeviceSynchronize(); printf("<----->\n");
				gnn_kernels::printGPU(devB,n,k);
				cudaDeviceSynchronize(); printf("<----->\n");
				for(int x = 0; x<m * k;x++){
					printf("%.4f ", hostD[x]);
					if((x+1)%k==0) printf("\n");
				}
			}else{
				for(int x = 0; x<nlayer * bsize;x++){
					if(((hostD[x] - hostC[x]) > 0.001 )){
						printf("Result matrices do not match(%f,%f)!!!\n",hostD[x],hostC[x] );
					}
				}
			}
			cudaFreeHost(hostD);*/
		}else if(test == TMMUL){
			// devB = devA * devC
			// (n x k) = (m x n) (m x k) <=> (n x k) = (m x n)^T (m x k) <=> (n x k) = (n x m) (m x k)
			Time<millis> t;
			dim3 agrid((bsize - 1)/TTILE + 1, (clayer - 1)/TTILE + 1);
			dim3 ablock(TTILE,TTILE);
			t.start();
			gnn_kernels::tmmul<DATA_T,TTILE><<<agrid,ablock>>>(
					devB,//n
					devA,//
					devC,//
					clayer,
					nlayer,
					bsize
			);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing tmmul kernel");
			t.lap("GPU serial tmmul elapsed time");

			allocHostMem<DATA_T>(&hostD,sizeof(DATA_T) * clayer * bsize, "Error allocating devC memory");
			safeCpyToHost<DATA_T>(hostA,devA,sizeof(DATA_T)*nlayer*clayer,"Error copying devA to host");
			safeCpyToHost<DATA_T>(hostB,devB,sizeof(DATA_T)*clayer*bsize,"Error copying devB to host");
			safeCpyToHost<DATA_T>(hostC,devC,sizeof(DATA_T)*nlayer*bsize,"Error copying devC to host");

			/*t.start();
			for(int x = 0; x < clayer; x++){//3
				for (int y = 0; y < bsize; y++){//3
					hostD[x * bsize + y] = 0.0;
					for (int z = 0; z < nlayer; z++){//2
						hostD[x * bsize + y] += hostA[z * clayer + x] * hostC[z * bsize + y];
					}
				}
			}
			if(!debug) t.lap("CPU serial mmul elapsed time");

			if(debug){
				//print_grid(agrid,ablock);
				//gnn_kernels::printGPU(devA,nlayer,clayer);
				//cudaDeviceSynchronize(); printf("<----->\n");
				//gnn_kernels::printGPU(devC,nlayer,bsize);
				//cudaDeviceSynchronize(); printf("<----->\n");
				gnn_kernels::printGPU(devB,clayer,bsize);
				cudaDeviceSynchronize();
				for(int x = 0; x<clayer * bsize;x++){
					printf("%.4f ", hostD[x]);
					if((x+1)%k==0) printf("\n");
				}
			}else{
				for(int x = 0; x<clayer * bsize;x++){
					if(((hostD[x] - hostB[x]) > 0.001 )){
						printf("Result matrices do not match(%f,%f)!!!\n",hostD[x],hostB[x] );
					}
				}
			}*/
			cudaFreeHost(hostD);
		}else if (test == MHPROD){
			dim3 dgrid((bsize - 1) / TTILE + 1, (clayer - 1) / TTILE + 1);
			dim3 dblock(TTILE, TTILE);

			if(debug){
				printf("D=");
				gnn_kernels::printGPU(devC,nlayer,bsize);
				cudaDeviceSynchronize();
				printf("W=");
				gnn_kernels::printGPU(devA,nlayer,clayer);
				cudaDeviceSynchronize();
				printf("A=");
				gnn_kernels::printGPU(devB,clayer-1,bsize);
				cudaDeviceSynchronize();
			}
			Time<millis> t;
			t.start();
			gnn_kernels::hmprod_mmul<DATA_T,ACT_F,TTILE><<<dgrid, dblock>>>(
					devC,
					devA,
					devB,
					F,
					nlayer,
					clayer-1,
					bsize
					);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing hmprod_tmmul kernel");
			t.lap("GPU serial hmprod elapsed time");

			allocHostMem<DATA_T>(&hostD,sizeof(DATA_T) * nlayer * bsize, "Error allocating devC memory");
			safeCpyToHost<DATA_T>(hostA,devA,sizeof(DATA_T)*nlayer*clayer,"Error copying devA to host");
			safeCpyToHost<DATA_T>(hostB,devB,sizeof(DATA_T)*clayer*bsize,"Error copying devB to host");
			safeCpyToHost<DATA_T>(hostC,devC,sizeof(DATA_T)*nlayer*bsize,"Error copying devC to host");

			/*for(int x = 0; x < nlayer; x++){//3
				for (int y = 0; y < bsize; y++){//3
					hostD[x * bsize + y] = hostA[x * (clayer) + clayer - 1];
					for (int z = 0; z < clayer - 1; z++){//2
						hostD[x * bsize + y] += hostA[x * (clayer) + z] * hostB[z * bsize + y];
					}
					hostD[x * bsize + y] = F.f(hostD[x * bsize + y]);
				}
			}*/

			if(debug){
				printf("R=");
				gnn_kernels::printGPU(devC,nlayer,bsize);
				cudaDeviceSynchronize();
				for(int x = 0; x<nlayer * bsize;x++){
					printf("%.4f ", hostD[x]);
					if((x+1)%k==0) printf("\n");
				}
			}
		}else if( test == TVECPVEC ){
			dim3 grid((clayer - 1)/TTILE + 1, (nlayer - 1)/TTILE + 1 );
			dim3 block(TTILE,TTILE);

			if(false){
				//print_grid(grid,block);
				printf("W=");
				gnn_kernels::printGPU(devA,nlayer,clayer);
				cudaDeviceSynchronize();
				printf("D=");
				gnn_kernels::printGPU(devC,nlayer,bsize);
				cudaDeviceSynchronize();
				printf("A=");
				gnn_kernels::printGPU(devB,clayer-1,bsize);
				cudaDeviceSynchronize();
				printf("E=tvecpvec(W,D,A,%f,%d)\n",0.3233,bsize);
			}

			Time<millis> t;
			t.start();
			gnn_kernels::tvecpvec<DATA_T,TTILE><<<grid,block>>>(
					devA,
					devC,
					devB,
					nlayer,
					bsize,
					clayer-1,
					0.0231
					);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing tvecpvec kernel");
			t.lap("GPU serial tvecpvec elapsed time");

			if(debug){
				//printf("R=");
				gnn_kernels::printGPU(devA,nlayer,clayer);
				cudaDeviceSynchronize();
				//printf("round(R-E)\n");
			}
		}

		cudaFree(devA); cudaFree(devB); cudaFree(devC);
		cudaFreeHost(hostA); cudaFreeHost(hostB); cudaFreeHost(hostC); cudaFreeHost(hostD);
		cudaDeviceReset();
	}

	template class GNeuralNetwork<float,gnn_actf::Sigmoid>;
	template class GNeuralNetwork<float,gnn_actf::FSigmoid>;
	template class GNeuralNetwork<float,gnn_actf::Arctan>;

	//template class GNeuralNetwork<double,gnn_actf::Sigmoid>;
	//template class GNeuralNetwork<double,gnn_actf::FSigmoid>;
	//template class GNeuralNetwork<double,gnn_actf::Arctan>;
}
