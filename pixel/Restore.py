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
import glob

flags = tf.app.flags
conf = flags.FLAGS
# imgs_list_path="data/train_small.txt"
model_path="04-22_models/model.ckpt-50000"
#model_path="restore_model/model.ckpt-BEST"
sample_path="sample_images/4/*.jpg"
#sample_path="new_gifs_frames/*.jpg"
# sample_path="gif_frames/*.jpg"
gen_path="04-26_samples/"

class Restore(object):
  def __init__(self):
    self.sz_hr = 32
    self.sz_lr= 8
    self.batch_size = 32
    self.SampleSet=glob.glob(sample_path)
    self.SampleSet.sort(key=lambda f: int(filter(str.isdigit, f)))
    #print(self.SampleSet)
    self.dict=self.get_dict(self.SampleSet)
    #print(self.dict)
    print("number of sample images: %d"%len(self.SampleSet))
    device_str="/gpu:0"
    with tf.device(device_str):
      #dataset
      # self.dataset = DataSet(imgs_list_path, 15, self.batch_size,self.sz_hr,self.sz_lr,False)
      self.dataset=feed_dataset(self.SampleSet)
      self.net = Net(self.dataset.hr_images, self.dataset.lr_images, 'prsr',self.sz_hr,self.sz_lr)
  def get_dict(self,path):
    dict={}
    for file in path:
      arr=file.split('/')[-1].split('-')
      dict[file]=str(arr[0])
    return dict
  def restore_image(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    
    print("Reloading model...")
    saver.restore(sess, model_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # loss = sess.run(self.net.loss, feed_dict={self.net.train: False})
    # print("loss is: %.3f" %loss)
    print("Start Generating...")
    path=gen_path
    if not os.path.exists(path):
      #shutil.rmtree(path)
      os.mkdir(path)
    
    #print (self.SampleSet)
    
    for i in xrange(int(len(self.SampleSet)/32)):
      hr_feed,lr_feed=self.sample_data(i)
      self.sample(sess, mu=1.1, hr_feed=hr_feed,lr_feed=lr_feed,batchNum=i,step=0,path=path,feed=True,gen=True)
    
    coord.request_stop()
    coord.join(threads)
    sess.close()
  def sample_data(self,batchNum):
    '''
    Output: High resolution and low resolutoin images as input data.
    '''
    hr_feed=np.zeros([self.batch_size,self.sz_hr,self.sz_hr,3])
    lr_feed=np.zeros([self.batch_size,self.sz_lr,self.sz_lr,3])
    for i in range(self.batch_size):
      img=Image.open(self.SampleSet[i+batchNum*32])
      hr_feed[i]=np.asarray(img.resize((32,32),Image.ANTIALIAS))
      lr_feed[i]=np.asarray(img.resize((8,8),Image.ANTIALIAS))
    return hr_feed,lr_feed

  # Sampling function, decide FEED and GEN.
  def sample(self, sess, hr_feed,lr_feed,batchNum,mu=1.1, step=None,path=None,feed=False,gen=True):
    '''
    Input: Session, step, path, feed, gen
    Output: If feed==True, feed data from sample_data(), otherwise will use random data; 
            if gen==False, don't generate images, just return input hr_images and lr_images.
    '''
    print("Current batch: %d"%batchNum)
    c_logits = self.net.conditioning_logits
    p_logits = self.net.prior_logits
    lr_imgs = self.dataset.lr_images
    hr_imgs = self.dataset.hr_images
    np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
    #np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs],feed_dict={hr_imgs:hr_feed,lr_imgs:lr_feed})
    gen_hr_imgs = np.zeros((self.batch_size, self.sz_hr, self.sz_hr, 3), dtype=np.float32)
    np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.net.train:False})

    tag1='/gen'
    tag2='/prior'
    tag3='/condition'
    tag4='/hr'
    tag5='/lr'
    file_names=self.SampleSet[batchNum*32:(batchNum+1)*32]
    file_pathes=[]
    for file in file_names:
      file_pathes.append(self.dict[file])
    #
    for i,path in enumerate(file_pathes):
      path=gen_path+path
      file_pathes[i]=path
      if not os.path.exists(path+tag1):
        os.makedirs(path+tag1)
      if not os.path.exists(path+tag2):
        os.makedirs(path+tag2)
      if not os.path.exists(path+tag3):
        os.makedirs(path+tag3)
      if not os.path.exists(path+tag4):
        os.makedirs(path+tag4)
      if not os.path.exists(path+tag5):
        os.makedirs(path+tag5)
    #print(file_pathes)
    #rand=subpixelnize(np_hr_imgs)
    con_hr_imgs = np.zeros((self.batch_size, self.sz_hr, self.sz_hr, 3), dtype=np.float32)
    pri_hr_imgs = np.zeros((self.batch_size, self.sz_hr, self.sz_hr, 3), dtype=np.float32)
    #pri_hr_imgs[:,:,:,0]=np_hr_imgs[:,:,:,0]
    if gen :
      for i in range(self.sz_hr):
        for j in range(self.sz_hr):
          for c in range(3):
            np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs, self.net.train:False})
            new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256]+np_p_logits[:, i, j, c*256:(c+1)*256], mu=mu)
            #new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=1.1)
            gen_hr_imgs[:, i, j, c] = new_pixel

            con_pixel=logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=1.1)
            con_hr_imgs[:, i, j, c] = con_pixel


            #pri_pixel=logits_2_pixel_value(rand[:, i, j, c*256:(c+1)*256]+np_p_logits[:, i, j, c*256:(c+1)*256], mu=1.1)
            #pri_hr_imgs[:, i, j, c] = pri_pixel

        print ("current row: (%d)"%(i))
      self.save_samples_with_path(gen_hr_imgs, file_pathes,tag1)
    #self.save_samples_with_path(pri_hr_imgs, file_pathes,tag2)
    self.save_samples_with_path(con_hr_imgs, file_pathes,tag3)
    self.save_samples_with_path(np_hr_imgs, file_pathes,tag4)
    self.save_samples_with_path(np_lr_imgs, file_pathes,tag5)
  def save_samples_with_path(self,np_imgs,pathes,flag):
    #print(len(pathes))
    np_imgs = np_imgs.astype(np.uint8)
    N, H, W, I = np_imgs.shape
    num = int(N ** (0.5))+1
    merge_img = np.zeros((H, W, 3), dtype=np.uint8)
    idx=0
    for i in range(num):
      for j in range(num):
        frame=i*num+j
        if frame<N:
          idx=existing_len(pathes[frame]+flag)
          merge_img[:,:,:] = np_imgs[frame,:,:,:]
          imsave(pathes[frame]+flag+'/'+str(idx).zfill(3)+".jpg", merge_img)
def subpixelnize(images):
    n_batch,n_row,n_col,chanel=np.shape(images)
    result=3*np.random.randn(32, 32, 32, 3*256).astype(np.float32)
    for i in range(n_batch):
        for j in range(n_row):
            for k in range(n_col):
                for l in range(chanel):
                  result[i][j][k][l*256:(l+1)*256][int(images[i][j][k][l])]+=3

    return result
def existing_len(folder_name):
    list=os.listdir(folder_name)
    return len(list)
def main(_):
  restore = Restore()
  restore.restore_image()
if __name__ == '__main__':
  tf.app.run()