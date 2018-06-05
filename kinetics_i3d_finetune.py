from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse

from data_utils.imagenet_data import ImagenetData
from data_utils.op_processing import distorted_inputs

from utils_input import *

import i3d
import os
import sonnet

_SAMPLE_VIDEO_FRAMES = 15
_IMAGE_SIZE = 224
_NUM_CLASSES = 51
_EPOCHS = 10
_BATCH_SIZE = 4

TRAINING = True

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

    # Calculate the learning rate schedule.
    num_steps_per_epoch = (dataset._num_train //
                            FLAGS.batch_size // FLAGS.num_gpus)


    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
      rgb_net, _ = rgb_model(images_placeholder, is_training=False, dropout_keep_prob=1.0)
      end_point = 'Logits'
      with tf.variable_scope(end_point):
        rgb_net = tf.nn.avg_pool3d(rgb_net, ksize=[1, 2, 7, 7, 1],
                               strides=[1, 1, 1, 1, 1], padding=sonnet.VALID)
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

        print("averaged_logits")
        print(averaged_logits)

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

    print("model_predictions")
    print(model_predictions)


    
    config = tf.ConfigProto(allow_soft_placement = True)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        print('\033[91m' + "Total number of parameters: " +
            str(total_parameters) + '\033[0m')  

        # ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder

        # flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())        
        
        # print('\033[91m' + "The number of flops: " +
        #     str(flops) + '\033[0m')  
        # Initializing Variables
        sess.run(init_op)

        for step in xrange(200*num_steps_per_epoch):
                if change_lrn_rate:
                    if step < lrn_rate_change[0] * num_steps_per_epoch:
                        lrn_rate = FLAGS.init_lrn_rate
                    elif step < lrn_rate_change[1] * num_steps_per_epoch:
                        lrn_rate = 0.1 * FLAGS.init_lrn_rate
                    elif step < lrn_rate_change[2] * num_steps_per_epoch:
                        lrn_rate = 0.01 * FLAGS.init_lrn_rate
                    elif step < lrn_rate_change[3] * num_steps_per_epoch:
                        lrn_rate = 0.001 * FLAGS.init_lrn_rate
                    else:
                        lrn_rate = 0.0001 * FLAGS.init_lrn_rate   

                # Run training step
                train_image_batch_arr, train_label_batch_arr = sess.run([train_image_batch, train_label_batch])
                start_time = time.time()
            
                _, loss_value, train_acc, top_5_acc, t_c_e, r_l, lolz = sess.run([train_op, loss, mean_train_accuracy, mean_top_5_train, tce, rl, lololol], feed_dict={lr: lrn_rate,
                    images_placeholder: train_image_batch_arr, labels_placeholder: train_label_batch_arr}) 
                
                duration += time.time() - start_time
               
                if step %10 ==0:

                    print("step " +str(step)+" train acc: "+str(train_acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HMDB51',
                                            help='Dataset:  ["HMDB51"] ')
    parser.add_argument('--filename', type=str, default=None,
                                            help='Filename of the dataset. [None]')
    parser.add_argument('--train_data_percent', type=float, default=1,
                                            help='The percentage of subsample of the training data. [1]')
    parser.add_argument('--batch_size', type=int, default=4,
                                            help='Batch size. [4]')
    parser.add_argument('--max_steps', type=int, default=100000,
                                            help='Max number of training steps. [20000]')
    parser.add_argument('--init_lrn_rate', type=float, default=0.001,
                                            help='Initial learning rate. [0.001]')
    parser.add_argument('--lrn_rate_change', type=str, default="80, 120, 160, 200",
                                            help='A list of number of epochs when the learning rate is reduced by a factor of 10. The learning rate is reduced three times. For example, if lrn_rate_change=[80, 120, 160], then the learning rate is reduced at 50/70/90 epochs. [50, 70, 90]')
    parser.add_argument('--seq_len', type=int, default=15,
                                            help='seqlength. [16]')
    parser.add_argument('--hp_weight_decay_rate', type=float, default=0.0002,
                                            help='Weight_decay for L2 loss. [0.0002]')
    parser.add_argument('--hp_weight_smoothness_rate', type=float, default=0.0002,
                                            help='Weight smoothness rate for weight_smoothness function. [0.0002]')
    parser.add_argument('--log_dir_prefix', type=str, default='./log',
                                            help='Log directory. ["./log"]')
    parser.add_argument('--model_dir_prefix', type=str, default='./model',
                                            help='Model directory. ["./model"]')
    parser.add_argument('--restore_from_model', type=bool, default=False, dest='restore_from_model',
                                            help='Restore from the previous trained model. [False]')
    parser.add_argument('--log_freq', type=int, default=1,
                                            help='Frequency of printing training accuracy in epochs. By default, print every 1 epoch. [1]')
    parser.add_argument('--test_batch_size', type=int, default=4,
                                            help='Test batch size used to evaluate test accuracy. [8]')
    parser.add_argument('--num_gpus', type=int, default=1,
                                            help='Number of gpus to use')

   
   
    parser.set_defaults(data_augment=True, use_bn=True, use_multiscale=False, ms_scale_weights=True, display_top_5=False, dataset_stand=True)                        
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        raise Exception('Unknown arguments:' + ', '.join(unparsed))
    print(FLAGS)
    main()
     
