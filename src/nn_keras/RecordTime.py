from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings
from keras.callbacks import Callback
import time

class RecordTime(Callback):
    '''Record time after every epoch.
    # Arguments
        None
    '''
    def __init__(self):
        super(Callback, self).__init__()
        self.start_time = -1.0
        self.end_time = -1.0
        self.nb_epochs = 0.0
        self.tot_time = 0.0
        self.avg_time = 0.0

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.end_time = time.time()
        print('\nTime taken by epoch {0}: {1}\n'.format(epoch, self.end_time - self.start_time))
        self.nb_epochs += 1
        self.tot_time += (self.end_time - self.start_time)

    def on_train_begin(self, logs={}):
        self.start_time = -1.0
        self.end_time = -1.0
        self.nb_epochs = 0.0
        self.tot_time = 0.0
        self.avg_time = 0.0

    def on_train_end(self, logs={}):
        self.avg_time = self.tot_time/self.nb_epochs
        print('\nAverage time per epoch: {0}\n'.format(self.avg_time))
        