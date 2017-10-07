#first make

CC = g++
NVCC = nvcc
EXEC = common

LIB_FLAGS = -lcuda -lcudart
INCLUDE_PATHS = /usr/local/cuda/include
INCUDE_LIB = /usr/local/cuda-7.5/lib64

NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true

all: compile

compile:
	$(CC) -std=c++11 -c *.cpp -L $(INCLUDE_LIB) $(LIB_FLAGS) -I$(INCLUDE_PATHS)
	$(NVCC)  -std=c++11 $(NVCC_FLAGS) -c *.cu
	$(NVCC) -std=c++11 -gencode arch=compute_35,code=sm_35 *.o -o $(EXEC)
	
clean:
	rm *.o
	rm $(EXEC)
