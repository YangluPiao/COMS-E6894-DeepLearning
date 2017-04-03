from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
from data import *
from net import *
from utils import *
from PIL import Image
import os
import time
import shutil

flags = tf.app.flags
conf = flags.FLAGS

class Solver(object):
  def __init__(self):
    self.sz_hr = conf.size_hr
    self.sz_lr= conf.size_lr
    self.file_length=conf.file_length
    print("Size of hr images:%d*%d, lr images:%d*%d"%(self.sz_hr,self.sz_hr,self.sz_lr,self.sz_lr))
    self.device_id = conf.device_id
    self.train_dir = conf.train_dir
    self.samples_dir = conf.samples_dir
    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)
    if not os.path.exists(self.samples_dir):
      os.makedirs(self.samples_dir)
    #datasets params
    self.num_epoch = conf.num_epoch
    self.batch_size = conf.batch_size
    #optimizer parameter
    self.learning_rate = conf.learning_rate
    if conf.use_gpu:
      device_str = '/gpu:' +  str(self.device_id)
    else:
      device_str = '/cpu:0'
    with tf.device(device_str):
      #dataset
      self.dataset = DataSet(conf.imgs_list_path, self.num_epoch, self.batch_size,self.sz_hr,self.sz_lr)
      self.net = Net(self.dataset.hr_images, self.dataset.lr_images, 'prsr',self.sz_hr,self.sz_lr)
      #optimizer
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           500000, 0.5, staircase=True, name='lr_expDecay')
      optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8, name='RMSopt')
      self.train_op = optimizer.minimize(self.net.loss, global_step=self.global_step, name='RMSmin')
  def train(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    #saver.restore(sess, './models/model.ckpt-30000')
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    iters = 0
    print("Start training...")
    new_best=1000
    count=1
    try:
      while not coord.should_stop():
        # Run training steps or whatever
        t1 = time.time()
        _, loss = sess.run([self.train_op, self.net.loss], feed_dict={self.net.train: True})
        t2 = time.time()
        
        if loss<new_best:
          new_best=loss
          print('<---NEW BSET--->')
          print('step %d, loss = %.2f, %d steps since last best loss' % ((iters, loss, count)))
          print('<-------------->')
          count=1
        else:
          count+=1
          print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))
        if iters % 1000 == 1:
          summary_str = sess.run(summary_op, feed_dict={self.net.train: True})
          summary_writer.add_summary(summary_str, iters)
        # Sample images
        
        if iters % 5000 == 0 and iters > 1:
          path="./samples/"+str(iters)
          if os.path.exists(path):
            shutil.rmtree(path)
          os.mkdir(path)
          self.sample(sess, mu=1.1, step=iters,path=path,feed=False,gen=True)
        
        # Checkpoints
        if iters % 10000 == 0:
          checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=iters)
        if count % 1000 == 0 and count > 0:
          print("No improvement for %d steps"%(count))
        if count>50000:
          coord.request_stop()
        iters += 1
    except tf.errors.OutOfRangeError:
      checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path)
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
  def sample_data(self):
    '''
    Output: High resolution and low resolutoin images as input data.
    '''
    hr_feed=np.zeros([self.batch_size,self.sz_hr,self.sz_hr,3])
    lr_feed=np.zeros([self.batch_size,self.sz_lr,self.sz_lr,3])
    for i in range(self.batch_size):
      hr_feed[i,:,:,:]=np.asarray(Image.open("./sample_images/hr_"+str(i)+".jpg"))
      lr_feed[i,:,:,:]=np.asarray(Image.open("./sample_images/lr_"+str(i)+".jpg"))
    return hr_feed,lr_feed

  # Sampling function, decide FEED and GEN.
  def sample(self, sess, mu=1.1, step=None,path=None,feed=False,gen=True):
    '''
    Input: Session, step, path, feed, gen
    Output: If feed==True, feed data from sample_data(), otherwise will use random data; 
            if gen==False, don't generate images, just return input hr_images and lr_images.
    '''
    c_logits = self.net.conditioning_logits
    p_logits = self.net.prior_logits
    lr_imgs = self.dataset.lr_images
    hr_imgs = self.dataset.hr_images
    if feed :
      hr_feed,lr_feed=self.sample_data()
      np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs],feed_dict={hr_imgs:hr_feed,lr_imgs:lr_feed})
    else:
      np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
    gen_hr_imgs = np.zeros((self.batch_size, self.sz_hr, self.sz_hr, 3), dtype=np.float32)
    np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.net.train:False})
    print('Sampleing at iter %d'% (step))
    # f=open("c_logits/np_c_logits_"+str(step)+".txt",'w')
    # print (np_c_logits,file=f)
    # f.close()
    
    if gen :
      for i in range(self.sz_hr):
        for j in range(self.sz_hr):
          for c in range(3):
            np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
            new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256] + np_p_logits[:, i, j, c*256:(c+1)*256], mu=mu)
            # new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=1.1)
            gen_hr_imgs[:, i, j, c] = new_pixel
        print ("current: (%d)"%(i))
      # g=open("p_logits/np_p_logits_"+str(step)+".txt",'w')
      # print (np_p_logits,file=g)
      # g.close()
      save_samples(gen_hr_imgs, path + '/gen_')
    save_samples(np_lr_imgs, path + '/lr_')
    save_samples(np_hr_imgs, path + '/hr_')
    
