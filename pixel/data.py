from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DataSet(object):
  def __init__(self, images_list_path, num_epoch, batch_size,sz_hr,sz_lr,shuffle=True):
    # filling the record_list
    input_file = open(images_list_path, 'r')
    self.record_list = []
    for line in input_file:
      line = line.strip()
      self.record_list.append(line)
    filename_queue = tf.train.string_input_producer(self.record_list,num_epochs=num_epoch,shuffle=shuffle)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    
    # image_file = tf.placeholder(dtype=tf.string)
    self.image = tf.image.decode_jpeg(image_file, 3)
    #preprocess
    hr_image = tf.image.resize_images(self.image, [sz_hr, sz_hr])
    lr_image = tf.image.resize_images(self.image, [sz_lr, sz_lr])
    hr_image = tf.cast(hr_image, tf.float32)
    lr_image = tf.cast(lr_image, tf.float32)
    #
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 400 * batch_size
    self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
class feed_dataset(object):
  def __init__(self,SampleSet):
    filename_queue = tf.train.string_input_producer(SampleSet,shuffle=False)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file, 3)
    hr_image = tf.image.resize_images(image, [32, 32])
    lr_image = tf.image.resize_images(image, [8, 8])
    hr_images = tf.cast(hr_image, tf.float32)
    lr_images = tf.cast(lr_image, tf.float32)
    self.hr_images,self.lr_images=tf.train.batch([hr_images,lr_images],batch_size=32)