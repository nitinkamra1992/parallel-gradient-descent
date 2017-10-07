#ifndef DATALIB_H
#define DATALIB_H

extern const int NUM_CLASSES;
extern const int NUM_TRAIN;
extern const int NUM_TEST;
extern const int NUM_VAL;
extern const int NUM_FEATURES;

extern float **x_train, **y_train, **x_test, **y_test, **x_val, **y_val;

void readData(bool normalize = false);

void dataset_mem_alloc();

#endif // DATALIB_H