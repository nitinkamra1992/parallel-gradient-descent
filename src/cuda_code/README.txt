This is the CUDA C++ implementation of neural network training.

<Compilation Instructions>
1. To compile just execute make
2. Make sure you have configured all the required paths for the CUDA API.

<Execution Instructions>
In main there are two neural networks with 1 and 2 hidden layers respectively.
In order to execute the training of one of these neural networks you need to provide the input and output number of neurons,
the number of iterations for training, the batch size, the train and testing data files with a single example in each row.
To get more information about execution instructions submit ./snn without any arguments to get a menu of the execution
options.


A typical execution is as follows
./snn -i=100 -md=1 -n=0.1 -f1=mnist_train.csv -f2=mnist_test.csv -b=100 -in=784 -on=10
-i=100, run for 100 epochs
-md=1, choose first neural network
-n=0.1, neural network learning rate
-f1,-f2, train and test data file
-b=100, 100 training examples in a batch
-in=784, number of input neurons
-on=10, number of output neurons

Notes:
1. The training and test data must be a csv file. Each row should contain input neurons + output neurons values.
The input values should be first followed by the expected output.
2. Enabling GPU boost can increase the number of GFLOPS achieved through training.
3. -md=0 executes a benchmark on the kernels used for training. Use nvprof for more accurate execution timings.
4. There exist default values for number of iterations(50), learning rate(0.1) and batch size(100) so you need not specify
if you want.