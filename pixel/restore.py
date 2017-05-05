from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from PIL import Image
from utils import *
from data import *
# from net import *
import glob
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

batch_size=32
sz_hr=32
sz_lr=8

def sample(sess, c_logits,p_logits, hr,lr,train,gen=True):
  '''
  Input: Session, c_logits, p_logits, hr, lr, train. All tansors are got from restored model.
  Output: hr_images,lr_images,gen_images
  '''
  c_logits = c_logits
  p_logits = p_logits
  hr_imgs = hr
  lr_imgs = lr
  print('Generating samples...')
  hr_feed,lr_feed=sample_data()
  
  
  if gen:
    gen_hr_imgs = np.zeros((batch_size, sz_hr, sz_hr, 3), dtype=np.float32)
    np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: lr_feed/255.0, train:False})
    for i in range(sz_hr):
      for j in range(sz_hr):
        for c in range(3):
          np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs/255.0})
          new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256] + np_p_logits[:, i, j, c*256:(c+1)*256], mu=1.1)
          # new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=1.1)
          gen_hr_imgs[:, i, j, c] = new_pixel
      print ("current: (%d)"%(i))
    save_samples(gen_hr_imgs, 'sample_images/2_me_output/gen_')
    # save_samples(hr_imgs, 'restore_outputs/p/gen_')
  save_samples(lr_feed, 'sample_images/2_me_output/lr_')
  save_samples(hr_feed, 'sample_images/2_me_output/hr_')
def sample_data():
    '''
    Output: High resolution and low resolutoin images as input data.
    '''
    hr_feed=np.zeros([32,32,32,3])
    lr_feed=np.zeros([32,8,8,3])
    SampleSet=glob.glob("sample_images/2/*.jpg")
    SampleSet.sort(key=lambda f: int(filter(str.isdigit, f)))
    for i in range(32):
      img=Image.open(SampleSet[i])
      hr_feed[i]=np.asarray(img.resize((32,32),Image.ANTIALIAS))
      lr_feed[i]=np.asarray(img.resize((8,8),Image.ANTIALIAS))
    return hr_feed,lr_feed
def restore_images(model_path):
  with tf.device('/gpu:0'):

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    new_saver = tf.train.import_meta_graph(model_path+'.meta')
    new_saver.restore(sess, model_path)
    g=tf.get_default_graph()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("Preparing Dataset...")
    dataset = DataSet("sample_data.txt", 5, batch_size, sz_hr, sz_lr)
    
    c_logits=g.get_tensor_by_name('prsr/conditioning/conv/BiasAdd:0')
    p_logits=g.get_tensor_by_name('prsr/prior/concat:0')
    hr=g.get_tensor_by_name('prsr/truediv:0')
    lr=g.get_tensor_by_name('prsr/truediv_1:0')
    train=g.get_tensor_by_name('prsr/netTrainBool:0')
    sample(sess,c_logits,p_logits,hr,lr,train,gen=True)
if __name__ == '__main__':
  restore_images("restore_model/model.ckpt-BEST")
