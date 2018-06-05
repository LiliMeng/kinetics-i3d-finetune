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
import time

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

    labels_placeholder = tf.placeholder(tf.int64, shape=(_BATCH_SIZE))

    learning_rate = tf.placeholder(tf.float32, [])
    # Training or evaluation
    is_training = tf.placeholder(tf.bool, [])

    # Initialize regularization loss to zero.
    reg_loss = 0
    
    lrn_rate_change = list(map(int, FLAGS.lrn_rate_change.split(',')))

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

    train_logits = averaged_logits
    
    train_cross_entropy = tf.reduce_mean(train_logits)

    train_step = tf.train.MomentumOptimizer(
        learning_rate, FLAGS.hp_momentum).minimize(train_cross_entropy)
    correct_prediction = tf.equal(tf.argmax(train_logits, 1), tf.argmax(labels_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing all the variables
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a summary monitor to cross_entropy
    tf.summary.scalar('train_cross_entropy', train_cross_entropy)

    summary_op = tf.summary.merge_all()

    start_time = time.time()

    
    config = tf.ConfigProto(allow_soft_placement = True)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
     # Initializing all the variables
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a summary monitor to cross_entropy
    tf.summary.scalar('train_cross_entropy', train_cross_entropy)

    summary_op = tf.summary.merge_all()

    start_time = time.time()
    with tf.Session() as sess:

        total_parameters = 0
        for variable in tf.trainable_variables():
            print(variable.op.name)
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes

        print('\033[91m' + "Total number of parameters: " +
              str(total_parameters) + '\033[0m')      

        sess.run(init_op)

        log_dir = os.path.join(FLAGS.log_dir_prefix, FLAGS.model +
                               time.strftime("_%b_%d_%H_%M", time.localtime()))

        model_dir = os.path.join(FLAGS.model_dir_prefix, FLAGS.model +
                                 time.strftime("_%b_%d_%H_%M", time.localtime()))

        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)

        if tf.gfile.Exists(model_dir):
            tf.gfile.DeleteRecursively(model_dir)
        tf.gfile.MakeDirs(model_dir)

        if FLAGS.restore_from_model:
            # Restore the model from disk
            saver.restore(sess, FLAGS.restore_ckpt)
            print("Model restored from %s" % FLAGS.restore_ckpt)

        summary_writer = tf.summary.FileWriter(
            log_dir, graph=tf.get_default_graph())

        epoch = dataset.num_train() // FLAGS.batch_size
        best_test_acc = 0

        for i in range(FLAGS.max_steps):
            # Decrease the learning rate by a factor of 10, every
            # FLAGS.lrn_rate_change epochs.

            if i < lrn_rate_change[0] * epoch:
                lrn_rate = FLAGS.init_lrn_rate
            elif i < lrn_rate_change[1] * epoch:
                lrn_rate = 0.1 * FLAGS.init_lrn_rate
            elif i < lrn_rate_change[2] * epoch:
                lrn_rate = 0.01 * FLAGS.init_lrn_rate
            else:
                lrn_rate = 0.001 * FLAGS.init_lrn_rate

            
            step_start_time = time.time()
            train_image_batch_arr, train_label_batch_arr = sess.run([train_image_batch, train_label_batch])
            # Run training step
            train_step.run(feed_dict={
                images_placeholder: train_image_batch_arr, labels_placeholder: train_label_batch_arr, learning_rate: lrn_rate, is_training: True})

            #count the step duration
            #step_duration = time.time() - step_start_time
            #print("\033[1;32;40m Time per step:" + str(step_duration) + '\033[0m')


            if i % (epoch * FLAGS.log_freq) == 0:
                # Train the model, and also write summaries.
                # Every 100th step, measure test accuracy, and write test
                # summaries
                train_accuracy, summary = sess.run([accuracy, summary_op], feed_dict={
                    x: batch[0], y_: batch[1], learning_rate: lrn_rate, is_training: True})
                print('step %d, epoch %d \ntraining accuracy: %g' %
                      (i, i // epoch, train_accuracy))
                train_summ = tf.Summary()
                train_summ.value.add(tag='train_accuracy',
                                     simple_value=train_accuracy)
                summary_writer.add_summary(train_summ, i)
                summary_writer.add_summary(summary, i)

                # Evaluate test accuracy on all the test data
                test_acc = 0
                num_test_batches = data.num_test() // FLAGS.test_batch_size
                for test_batch_start in range(num_test_batches):
                    test_image_batch_arr, test_label_batch_arr = sess.run([test_image_batch, test_label_batch])
                    batch_test_acc = sess.run(accuracy, feed_dict={images_placeholder: test_image_batch_arr, labels: test_label_batch_arr,
                                                                   learning_rate: lrn_rate, is_training: False})
                    test_acc += batch_test_acc
                test_acc /= num_test_batches

                print('test accuracy: %g' % test_acc)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                    # Save the best model
                    ckpt_name = os.path.join(model_dir, "model_best.ckpt")
                    save_path = saver.save(sess, ckpt_name)
                    print("Model saved in file: %s" % save_path)

                print('\033[91m' + "best test accuracy: " +
                      str(round(best_test_acc, 6)) + '\033[0m')

                test_summ = tf.Summary()
                test_summ.value.add(tag='test_accuracy',
                                    simple_value=test_acc)
                summary_writer.add_summary(test_summ, i)
        

        summary_writer.close()

        print("The whole process is done! Write result to file result_log.md")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HMDB51',
                                            help='Dataset:  ["HMDB51"] ')
    parser.add_argument('--model', type=str, default='I3D',
                        help='Model: "I3D" ')
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
    parser.add_argument('--hp_momentum', type=float, default=0.9,
                        help='Momentum for MomentumOptimizer. [0.9]')
   
    parser.set_defaults(data_augment=True, use_bn=True, use_multiscale=False, ms_scale_weights=True, display_top_5=False, dataset_stand=True)                        
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        raise Exception('Unknown arguments:' + ', '.join(unparsed))
    print(FLAGS)
    main()
     
