from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.regularizers as Reg
from keras.optimizers import SGD
import cPickle, gzip
import numpy as np
from keras.utils import np_utils
from RecordTime import RecordTime

# seed the random number generator randomly
np.random.seed()

# Load the dataset
with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set = cPickle.load(f)

# Separate features and label
X_train, Y_train = train_set;
X_test, Y_test = test_set;
X_valid, Y_valid = valid_set;
N_train,D = X_train.shape
N_test,D = X_test.shape
N_valid,D = X_valid.shape

batch_size = 128
num_classes = 10
num_epoch = 500

y_train = np_utils.to_categorical(Y_train, num_classes)
y_test = np_utils.to_categorical(Y_test, num_classes)
y_valid = np_utils.to_categorical(Y_valid, num_classes)

model = Sequential()
model.add(Dense(1024, input_dim=D, init='glorot_uniform', W_regularizer=Reg.l2(l=0.0)))
model.add(Activation('sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(1024, init='glorot_uniform', W_regularizer=Reg.l2(l=0.0)))
model.add(Activation('sigmoid'))
# model.add(Dense(100, init='glorot_uniform', W_regularizer=Reg.l2(l=0.0001)))
# model.add(Activation('sigmoid'))
model.add(Dense(10, init='glorot_uniform', W_regularizer=Reg.l2(l=0.0)))
model.add(Activation('sigmoid'))
# model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=0.005, momentum=0.0, nesterov=False)
model.compile(loss='mean_squared_error',
# model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=["accuracy"])

# Define Callbacks
timerecorder = RecordTime()

# Training the model
model.fit(X_train, y_train, nb_epoch=num_epoch, batch_size=batch_size,
          validation_data = (X_valid, y_valid), verbose=1,
          callbacks=[timerecorder])
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

print('Test score: {0}, Test accuracy: {1}'.format(score[0], score[1]))