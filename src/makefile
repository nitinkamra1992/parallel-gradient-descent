#
#
# Makefile for CPU implementation of neural network
#
#

TARGET_SERIAL = sNNet sNNet_BLAS
TARGET_PARALLEL = pNNet pNNet_BLAS

FC       = gfortran
CC       = g++
FFLAGS   = -Wall
CFLAGS   = -Wall
CPLUSFLAGS = -std=c++11
DEBUGFLAGS = -g
OPTFLAGS = -O3
THREADFLAGS = -pthread

BLLIB  = -lblas

TARGETS = $(TARGET_SERIAL) $(TARGET_PARALLEL)

all : $(TARGETS)
	@echo Make Complete
	
sNNet : sNNet.cpp matrixop.cpp datalib.cpp matrixop.h datalib.h
	$(CC) $(CPLUSFLAGS) $(OPTFLAGS) sNNet.cpp matrixop.cpp datalib.cpp -lm $(BLLIB) -o sNNet

sNNet_BLAS :  sNNet_BLAS.cpp matrixop.cpp datalib.cpp matrixop.h datalib.h
	$(CC) $(CPLUSFLAGS) $(OPTFLAGS) sNNet_BLAS.cpp matrixop.cpp datalib.cpp -lm $(BLLIB) -o sNNet_BLAS
	
pNNet_BLAS : sNNet_BLAS.cpp matrixop.cpp datalib.cpp matrixop.h datalib.h
	$(CC) $(CPLUSFLAGS) $(OPTFLAGS) sNNet_BLAS.cpp matrixop.cpp datalib.cpp -lm $(BLLIB) -o pNNet_BLAS	

pNNet : pNNet.cpp matrixop.cpp datalib.cpp matrixop.h datalib.h
	$(CC) $(CPLUSFLAGS) $(OPTFLAGS) $(THREADFLAGS) pNNet.cpp matrixop.cpp datalib.cpp -lm $(BLLIB) -o pNNet
	
clean:
	rm -rf $(TARGETS)