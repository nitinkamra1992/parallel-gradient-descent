# Parallel-Gradient-Descent

@Author: Nitin Kamra
@Author: Palash Goyal
@Author: Sungyong Seo
@Author: Vasileios Zois

###############################################################################

Note that we performed the project on our own servers and not on HPCC, so to run the project on HPCC you need to write your own PBS scripts.

###############################################################################

To compile the code (assuming you are in project root directory):
1. Change directory to ./src/
2. Type "make" on the shell and press enter
3. Change directory to ./cuda/ next
4. Type "make" on the shell and press enter

###############################################################################

NOTE: We do not provide the MNIST dataset. It needs to be downloaded from here: 
http://yann.lecun.com/exdb/mnist/ 
for the following implementations and placed in the ./data folder

To run the naive serial/BLAS serial/Pthreads parallel/BLAS parallel code after compiling (assuming you are in project root directory):
1. Change directory to ./src/
2. Use <./sNNet> to run the naive serial implementation
   Use <export OPENBLAS_NUM_THREADS=1> followed by <./sNNet_BLAS> to run the BLAS serial implementation
   Use <./pNNet> to run the Pthreads parallel implementation
   Use <export OPENBLAS_NUM_THREADS=32> followed by <./pNNet_BLAS> to run the BLAS parallel implementation

###############################################################################

NOTE: We do not provide the MNIST dataset. It needs to be downloaded from here: 
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz 
for the following implementations and placed in the ./data folder

To run the keras serial/keras parallel/keras GPU code after compiling (assuming you are in project root directory):
1. Make sure you have the open-source libraries Theano and Keras set up (and all their dependencies)
2. Change directory to ./src/nn_keras/
   Use <mv theanorc_cpu ~/.theanorc> followed by <NUM_OMP_THREADS=1 python mlp_keras.py> to run the theano serial implementation
   Use <mv theanorc_cpu ~/.theanorc> followed by <python mlp_keras.py> to run the theano parallel implementation
   Use <mv theanorc_gpu ~/.theanorc> followed by <python mlp_keras.py> to run the theano GPU implementation

###############################################################################

NOTE: We do not provide the MNIST dataset. It needs to be downloaded from here: 
http://yann.lecun.com/exdb/mnist/ 
for the following implementations and placed in the ./data folder.
Next from the main project directory go to ./src/ and run <./convertData> (after compiling with make). 
This will convert the MNIST dataset to the required CSV format.

To run the cuda implementation after compiling (assume you are in project root directory):
1. Change the directory to ./src/cuda_code and follow the instructions in README.txt

###############################################################################