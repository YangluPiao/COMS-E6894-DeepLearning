from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

filename_queue = tf.train.string_input_producer(["sample_images/3/hr_31.jpg"])
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file, 3)

filename_queue2 = tf.train.string_input_producer(["sample_images/3/hr_30.jpg"])
image_reader2 = tf.WholeFileReader()
_, image_file2 = image_reader.read(filename_queue)
image2 = tf.image.decode_jpeg(image_file2, 3)

sess=tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
print(sess.run([image,image2]))
coord.request_stop()
coord.join(threads)
sess.close()