from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import scipy.io
import os
import re
import sys
import tarfile

from six.moves import urllib, xrange
import tensorflow as tf
from PIL import Image

class DataSet(object):
    """
    Loads data and Keeps track of dataset, data paths, image height/width, channels, number of data/classes 
    """
    def __init__(self, datadir, 
                batchsize, testbatchsize, data_percent=1, 
                augment=True, dataset="HMDB51", scaled_size=0, dataset_stand=True):

        assert datadir is not None
        assert batchsize is not None
        assert testbatchsize is not None

        if not os.path.exists(datadir):
            os.makedirs(datadir)
        
        self._dataset = dataset
        self._datadir = datadir
        self._data_percent = data_percent
        self._batchsize = batchsize
        self._test_batchsize = testbatchsize
        self._augment = augment
        self._rescale = not not scaled_size
        self._dataset_stand = dataset_stand
        
        if self._dataset == "UCF101":

            print("UCF101 RGB dataset is used now")

            self._height = 56
            self._width = 56
            self._channels = 3
            self._num_train = 9537
            self._num_test = 3780#3783  
            self._num_classes = 101
            self._padding = 0

        elif self._dataset == "HMDB51":
            
            print("HMDB51 RGB dataset is used now")
            
            self._height = 56
            self._width = 56
            self._channels = 3
            self._num_train = 3570
            self._num_test = 1530
            self._num_classes = 51
            self._padding = 0

        elif self._dataset == "Moments":

            print("MIT Moments RGB dataset is used now")

            self._height = 56
            self._width = 56
            self._channels = 3
            self._num_train = 802244
            self._num_test = 33900  
            self._num_classes = 339
            self._padding = 0

        else: 
            raise Exception("Dataset: %s has not been implemented yet. Please check spelling." % dataset)
  