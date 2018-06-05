# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts ImageNet data to TFRecords file format with Example protos.
The raw ImageNet data set is expected to reside in JPEG files located in the
following directory structure.
  data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
  data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
  ...
where 'n01440764' is the unique synset label associated with
these images.
The training data set consists of 1000 sub-directories (i.e. labels)
each containing 1200 JPEG images for a total of 1.2M JPEG images.
The evaluation data set consists of 1000 sub-directories (i.e. labels)
each containing 50 JPEG images for a total of 50K JPEG images.
This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.
  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024
and
  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128
Each validation TFRecord file contains ~390 records. Each training TFREcord
file contains ~1250 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:
  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'
  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/synset: string specifying the unique ID of the label,
    e.g. 'n01440764'
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'
  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.
Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
Running this script using 16 threads may take around ~2.5 hours on an HP Z420.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
import cv2

IMG_SIZE = 256

tf.app.flags.DEFINE_string('train_directory', '/tmp/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', './data/hmdb51_rgb_3d/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 5,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 5,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 5,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as a synset. We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
tf.app.flags.DEFINE_string('labels_file',
                           'imagenet_lsvrc_2015_synsets.txt',
                           'Labels file')

# This file containing mapping from synset to human-readable label.
# Assumes each line of the file looks like:
#
#   n02119247    black fox
#   n02119359    silver fox
#   n02119477    red fox, Vulpes fulva
#
# where each line corresponds to a unique mapping. Note that each line is
# formatted as <synset>\t<human readable label>.
tf.app.flags.DEFINE_string('imagenet_metadata_file',
                           'imagenet_metadata.txt',
                           'ImageNet metadata file')

# This file is the output of process_bounding_box.py
# Assumes each line of the file looks like:
#
#   n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
#
# where each line corresponds to one bounding box annotation associated
# with an image. Each line can be parsed as:
#
#   <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
#
# Note that there might exist mulitple bounding box annotations associated
# with an image file.
tf.app.flags.DEFINE_string('bounding_box_file',
                           './imagenet_2012_bounding_boxes.csv',
                           'Bounding box file')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'gray'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename[0])),
      'image/encoded1': _bytes_feature(image_buffer[0]),
      'image/encoded2': _bytes_feature(image_buffer[1]),
      'image/encoded3': _bytes_feature(image_buffer[2]),
      'image/encoded4': _bytes_feature(image_buffer[3]),
      'image/encoded5': _bytes_feature(image_buffer[4]),
      'image/encoded6': _bytes_feature(image_buffer[5]),
      'image/encoded7': _bytes_feature(image_buffer[6]),
      'image/encoded8': _bytes_feature(image_buffer[7]),
      'image/encoded9': _bytes_feature(image_buffer[8]),
      'image/encoded10': _bytes_feature(image_buffer[9]),
      'image/encoded11': _bytes_feature(image_buffer[10]),
      'image/encoded12': _bytes_feature(image_buffer[11]),
      'image/encoded13': _bytes_feature(image_buffer[12]),
      'image/encoded14': _bytes_feature(image_buffer[13]),
      'image/encoded15': _bytes_feature(image_buffer[14])}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})

    print(image.shape)
    assert len(image.shape) ==3
    assert image.shape[2] == 3
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  # File list from:
  # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
  return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
  """Determine if file contains a CMYK JPEG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
               'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
               'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
               'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
               'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
               'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
               'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
               'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
               'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
               'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
               'n07583066_647.JPEG', 'n13037406_4650.JPEG']
  return filename.split('/')[-1] in blacklist


def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Clean the dirty data.
  if _is_png(filename):
    # 1 image is a PNG.
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_cmyk(filename):
    # 22 JPEG images are in CMYK colorspace.
    print('Converting CMYK to RGB for %s' % filename)
    image_data = coder.cmyk_to_rgb(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, labels,  num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0


  if not tf.gfile.Exists(FLAGS.output_directory):
    tf.gfile.MakeDirs(FLAGS.output_directory)
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)

    output_file = os.path.join(FLAGS.output_directory, output_filename)
  
   
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      image_buffers = []
      for j in range(15):
        image_buffer, height, width = _process_image(filename[j], coder)
        image_buffers.append(image_buffer)

      example = _convert_to_example(filename, image_buffers, label, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames,  labels, num_shards):
  """Process and save list of images as TFRecord of Example protos.
  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    
    num_shards: integer number of shards for this data set.
  """

  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
           labels,num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(img_list_file):
  """Build a list of all images files and labels in the data set.
  Args:
    img_list_file: txt file containing list of image directory and labels
     
  Returns:
    filenames: list of strings; each string is a path to an image file.
    labels: list of integer; each integer identifies the ground truth.
  """
  with open(img_list_file) as f:
    lines_img = f.readlines()

  # list for storing data and label for all the videos

  all_imgs_rgb_filenames = []
  
  labels = []
  
  
  for i in range(len(lines_img)):
    path_img_rgb = []
 
    label_img = int(lines_img[i].split('\t')[1])

    labels.append(int(label_img))
    assert(int(label_img)!=-1)
    img_path_suffix = lines_img[i].split('\t')[0]
    img_names = os.listdir(img_path_suffix)

    if len(img_names)>70:
      start_idx = random.randint(1, (len(img_names)-70))

      for j in range(15):
          tmp = os.path.join(img_path_suffix, 'image_' + str('%05d'%(start_idx+j*5)) + '.jpg')
          print(tmp)
          path_img_rgb.append(tmp)
    else:
      print(len(img_names))
      skip_frame=len(img_names)//20
      start_idx = random.randint(1, (len(img_names)-skip_frame*15))
      for j in range(15):
          tmp = os.path.join(img_path_suffix, 'image_' + str('%05d'%(start_idx+j*skip_frame)) + '.jpg')
          print(tmp)
          path_img_rgb.append(tmp)

    all_imgs_rgb_filenames.append(path_img_rgb)

  assert(len(all_imgs_rgb_filenames)==len(labels))
  return all_imgs_rgb_filenames,  labels


def _process_dataset(name, img_list_file, num_shards):
  """Process a complete data set and save it as a TFRecord.
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  """
  filenames, labels = _find_image_files(img_list_file)

  #for i in range(10):
  _process_image_files(name, filenames, labels, num_shards)

def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  print('Saving results to %s' % FLAGS.output_directory)


  # Run it!
  train_img_file_list = "../data_list/HMDB51/img_list/train.list"
  test_img_file_list = "../data_list/HMDB51/img_list/test.list"
  _process_dataset('train', train_img_file_list, FLAGS.train_shards)
  _process_dataset('validation', test_img_file_list, FLAGS.test_shards)


if __name__ == '__main__':
  tf.app.run()