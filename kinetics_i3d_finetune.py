from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from data_utils.imagenet_data import ImagenetData
from data_utils.op_processing import distorted_inputs

from utils_input import *

import i3d
import os

_SAMPLE_VIDEO_FRAMES = 15
_IMAGE_SIZE = 224
_NUM_CLASSES = 51
_EPOCHS = 10
_BATCH_SIZE = 4

def _get_dataset_train(train_batch_size):
  """Prepares a dataset input tensors."""
  #num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpu
  num_preprocess_threads = 16
  dataset = ImagenetData(subset="train")
  images, labels = distorted_inputs(
      dataset,
      batch_size=train_batch_size,
      num_preprocess_threads=num_preprocess_threads)
  return images, labels

def _get_dataset_test(test_batch_size):
  """Prepares a dataset input tensors."""
  #num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpu
  num_preprocess_threads = 16
  dataset = ImagenetData(subset="validation")
  images, labels = distorted_inputs(
      dataset,
      batch_size=test_batch_size,
      num_preprocess_threads=num_preprocess_threads)
  return images, labels

def main():
    # Fetch data and create queues for reading images/labels
    dataset = DataSet(datadir="./data", batchsize=_BATCH_SIZE, 
                      testbatchsize=_BATCH_SIZE, dataset='HMDB51')

    images_placeholder = tf.placeholder(tf.float32, shape=(_BATCH_SIZE, 
                                                           _SAMPLE_VIDEO_FRAMES,
                                                           _IMAGE_SIZE,
                                                           _IMAGE_SIZE,
                                                           3,
                                                           ))

    labels_placeholder = tf.placeholder(tf.int64, shape=(_BATCH_SIZE, _NUM_CLASSES))

    train_image_batch, train_label_batch = _get_dataset_train(_BATCH_SIZE)
    test_image_batch, test_label_batch = _get_dataset_test(_BATCH_SIZE)

    with tf.variable_scope('RGB'):
      test_image_batch_arr, test_label_batch_arr = sess.run([test_image_batch, test_label_batch])
      rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
      rgb_net, _ = rgb_model(images_placeholder, is_training=False, dropout_keep_prob=1.0)
      end_point = 'Logits'
      with tf.variable_scope(end_point):
        rgb_net = tf.nn.avg_pool3d(rgb_net, ksize=[1, 2, 7, 7, 1],
                               strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        if TRAINING:
            rgb_net = tf.nn.dropout(rgb_net, 0.7)
        logits = i3d.Unit3D(output_channels=_NUM_CLASSES,
                        kernel_shape=[1, 1, 1],
                        activation_fn=None,
                        use_batch_norm=False,
                        use_bias=True,
                        name='Conv3d_0c_1x1')(rgb_net, is_training=True)

        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)

      # predictions = tf.nn.softmax(averaged_logits)


    rgb_variable_map = {}

    for variable in tf.global_variables():
        if variable.name.split("/")[-4] == "Logits": continue
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable

    #print(rgb_variable_map)
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    model_logits = averaged_logits
    model_predictions = tf.nn.softmax(model_logits)


main()