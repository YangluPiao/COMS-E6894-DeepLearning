from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from skimage import io

import tensorflow as tf
import numpy as np
import glob
import os
import shutil

path=glob.glob("./test_data/*.png")
store='test_images/'
# for idx,img_path in enumerate(path):
# 	# if os.path.exists('test.png'):
# 	# 	shutil.rmtree('test.png')
# 	img=io.imread(img_path)
# 	io.imsave("./test_images/"+str(idx)+".png",np.asarray(img))

if os.path.exists(store):
	shutil.rmtree(store)
	os.mkdir(store)
print (path)
filename_queue = tf.train.string_input_producer(path,shuffle=False) #  list of files to read
reader = tf.WholeFileReader()
_, image_file = reader.read(filename_queue)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)

	# Start populating the filename queue.

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(len(path)): #length of your filename list
		image = tf.image.decode_png(image_file,4) #here is your image Tensor :) 
		print("Image No.%d"%i)
		# img1=np.asarray(image.eval())
		# img=np.asarray(tf.image.resize_images(image,[32,32]).eval()).astype(np.uint8)
		img=np.asarray(tf.image.resize_images(image,[455,346]).eval()).astype(np.uint8)
		if i<10:
			io.imsave(store+"0000"+str(i)+".png",img)
			# io.imsave(store+"1000"+str(i)+".png",img)
		else:
			io.imsave(store+"000"+str(i)+".png",img)
			# io.imsave(store+"100"+str(i)+".png",img)

		coord.request_stop()
		coord.join(threads)
	