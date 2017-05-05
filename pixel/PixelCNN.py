from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ops import *
from utils import save_samples
from PIL import Image
import os
import time
import shutil
import tensorflow as tf

flags = tf.app.flags
conf = flags.FLAGS

class DataSet(object):
  def __init__(self, images_list_path, num_epoch, batch_size,sz_hr):
    # filling the record_list
    input_file = open(images_list_path, 'r')
    self.record_list = []
    for line in input_file:
      line = line.strip()
      self.record_list.append(line)
    filename_queue = tf.train.string_input_producer(self.record_list,num_epochs=num_epoch)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    
    # image_file = tf.placeholder(dtype=tf.string)
    self.image = tf.image.decode_jpeg(image_file, 3)
    #preprocess
    hr_image = tf.image.resize_images(self.image, [sz_hr, sz_hr])
    # lr_image = tf.image.resize_images(self.image, [sz_lr, sz_lr])
    hr_image = tf.cast(hr_image, tf.float32)
    # lr_image = tf.cast(lr_image, tf.float32)
    #
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 400 * batch_size
    self.hr_images= tf.train.shuffle_batch([hr_image], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
class Net(object):
  def __init__(self, hr_images, scope,sz_hr):
    """
    Args:[0, 255]
      hr_images: [batch_size, hr_height, hr_width, in_channels] float32
      lr_images: [batch_size, lr_height, lr_width, in_channels] float32
    """
    self.sz_hr=sz_hr

    with tf.variable_scope(scope) as scope:
      self.train = tf.placeholder(tf.bool, name="netTrainBool")
      self.construct_net(hr_images)
  def prior_network(self, hr_images):
    """
    Args:[-0.5, 0.5]
      hr_images: [batch_size, hr_height, hr_width, in_channels]
    Returns:
      prior_logits: [batch_size, hr_height, hr_width, 4*256]
    """
    with tf.variable_scope('prior') as scope:
      conv1 = conv2d(hr_images, self.sz_hr*2, [7, 7], strides=[1, 1], mask_type='A', scope="conv1")
      inputs = conv1
      state = conv1
      for i in range(20):
        inputs, state = gated_conv2d(inputs, state, [5, 5], scope='gated' + str(i))
      conv2 = conv2d(inputs, 1024, [1, 1], strides=[1, 1], mask_type='B', scope="conv2")
      conv2 = tf.nn.relu(conv2)
      prior_logits = conv2d(conv2, 3 * 256, [1, 1], strides=[1, 1], mask_type='B', scope="conv3")

      prior_logits = tf.concat([prior_logits[:, :, :, 0::3], prior_logits[:, :, :, 1::3], prior_logits[:, :, :, 2::3]], 3)
      
      return prior_logits
  def softmax_loss(self, logits, labels):
    logits = tf.reshape(logits, [-1, 256])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    return tf.losses.sparse_softmax_cross_entropy(
           labels, logits)
  def construct_net(self, hr_images):
    """
    Args: [0, 255]
    """
    #labels
    labels = hr_images
    #normalization images [-0.5, 0.5]
    hr_images = hr_images / 255.0 - 0.5
    self.prior_logits = self.prior_network(hr_images)

    loss = self.softmax_loss(self.prior_logits, labels)

    self.loss = loss
    tf.summary.scalar('loss', self.loss)

class Solver(object):
  def __init__(self):
    self.sz_hr = conf.size_hr
    self.file_length=conf.file_length
    print("Size of hr images:%d*%d"%(self.sz_hr,self.sz_hr))
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
      self.dataset = DataSet(conf.imgs_list_path, self.num_epoch, self.batch_size,self.sz_hr)
      self.net = Net(self.dataset.hr_images, 'prsr',self.sz_hr)
      #optimizer
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           20000, 0.5, staircase=True, name='lr_expDecay')
      optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8, name='RMSopt')
      self.train_op = optimizer.minimize(self.net.loss, global_step=self.global_step, name='RMSmin')
      self.SampleSet=os.listdir("sample_images")
  def train(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.trainable_variables())
    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    saver.restore(sess, './models/model.ckpt-30000')
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    iters = 0
    print("Start training...")
    new_best=1000
    count=1
    _, loss = sess.run([self.train_op, self.net.loss], feed_dict={self.net.train: True})
    iters=0
    print('Sampleing at iter %d'% (iters))
    path=self.samples_dir+'/'+str(iters)
    if os.path.exists(path):
      shutil.rmtree(path)
      os.mkdir(path)
    hr_feed=self.sample_data(0)
    self.sample(sess, mu=1.1, hr_feed=hr_feed,batchNum=0,step=iters,path=path,feed=True,gen=True)
    # try:
    #   while not coord.should_stop():
    #     # Run training steps or whatever
    #     t1 = time.time()
    #     _, loss = sess.run([self.train_op, self.net.loss], feed_dict={self.net.train: True})
    #     t2 = time.time()

    #     if loss<new_best:
    #       new_best=loss
    #       print('<---NEW BSET--->')
    #       print('step %d, loss = %.2f, %d steps since last best loss' % ((iters, loss, count)))
    #       print('<-------------->')
    #       count=1
    #     else:
    #       count+=1
    #       print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))
    #     if iters % 1000 == 0:
    #       summary_str = sess.run(summary_op, feed_dict={self.net.train: True})
    #       summary_writer.add_summary(summary_str, iters)

    #     # Sample images
    #     if iters % 5000 == 0 and iters>0:
    #       print('Sampleing at iter %d'% (iters))
    #       path=self.samples_dir+'/'+str(iters)
    #       if os.path.exists(path):
    #         shutil.rmtree(path)
    #       os.mkdir(path)
    #       # for i in range(162):
    #         # hr_feed,lr_feed=self.sample_data(i)
    #         # self.sample(sess, mu=1.1, hr_feed=hr_feed,lr_feed=lr_feed,batchNum=i,step=iters,path=path,feed=True,gen=False)
    #       hr_feed=self.sample_data(0)
    #       self.sample(sess, mu=1.1, hr_feed=hr_feed,batchNum=0,step=iters,path=path,feed=True,gen=True)
    #       # coord.request_stop()
        
    #     # Checkpoints
    #     if iters % 10000 == 0:
    #       checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
    #       saver.save(sess, checkpoint_path, global_step=iters)
    #     if count % 1000 == 0 and count > 0:
    #       print("No improvement for %d steps"%(count))
    #     if count>50000:
    #       coord.request_stop()
    #     iters += 1
    # except tf.errors.OutOfRangeError:
    #   checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
    #   saver.save(sess, checkpoint_path)
    #   print('Done training -- epoch limit reached')
    # finally:
    #   # When done, ask the threads to stop.
    #   coord.request_stop()

    # # Wait for threads to finish.
    # coord.join(threads)
    # sess.close()
  def sample_data(self,batchNum):
    '''
    Output: High resolution and low resolutoin images as input data.
    '''
    hr_feed=np.zeros([self.batch_size,self.sz_hr,self.sz_hr,3])
    for i in range(self.batch_size):
      img=Image.open("./sample_images/"+self.SampleSet[i+batchNum*32])
      hr_feed[i]=np.asarray(img.resize((32,32),Image.ANTIALIAS))
    # img=Image.open("./sample_images/"+self.SampleSet[i+batchNum*32])
    # hr_feed[i]=np.asarray(img.resize((32,32),Image.ANTIALIAS))
    return hr_feed

  # Sampling function, decide FEED and GEN.
  def sample(self, sess, hr_feed,batchNum,mu=1.1, step=None,path=None,feed=False,gen=True):
    '''
    Input: Session, step, path, feed, gen
    Output: If feed==True, feed data from sample_data(), otherwise will use random data; 
            if gen==False, don't generate images, just return input hr_images and lr_images.
    '''
    print("Current batch: %d"%batchNum)
    p_logits = self.net.prior_logits
    hr_imgs = self.dataset.hr_images
    # image = self.dataset.image
    if feed :
      np_hr_imgs= sess.run(hr_imgs,feed_dict={hr_imgs:hr_feed})
      # np_hr_imgs, np_lr_imgs=hr_feed,lr_feed
    else:
      np_hr_imgs = sess.run(hr_imgs)
    gen_hr_imgs = np.zeros((self.batch_size, self.sz_hr, self.sz_hr, 3), dtype=np.float32)
    origin=subpixelnize(np_hr_imgs)
    # for i in range(self.sz_hr):
    #     for j in range(self.sz_hr):
    #       for c in range(3):
    #         new_pixel = logits_2_pixel_value(np_c_logits[:, i, j, c*256:(c+1)*256], mu=mu)
    #         gen_hr_imgs[:, i, j, c] = new_pixel
    if gen :
      for i in range(self.sz_hr):
        for j in range(self.sz_hr):
          for c in range(3):
            np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
            # np_p_logits=binarize(np_p_logits)
            new_pixel = logits_2_pixel_value(origin[:, i, j, c*256:(c+1)*256]+np_p_logits[:, i, j, c*256:(c+1)*256], mu=mu)
            gen_hr_imgs[:, i, j, c] = new_pixel
        print ("current row: (%d)"%(i))
      save_samples(gen_hr_imgs, path + "/"+str(step)+'_gen_')
    save_samples(np_hr_imgs, path + "/"+str(step)+'_hr_')

def subpixelnize(images):
    n_batch,n_row,n_col,chanel=np.shape(images)
    result=np.random.randn(32, 32, 32, 3*256).astype(np.float32)
    for i in range(n_batch):
        for j in range(n_row):
            for k in range(n_col):
                for l in range(chanel):
                  result[i][j][k][l*256:(l+1)*256][int(images[i][j][k][l])]+=2

    return result
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference
def logits_2_pixel_value(logits, mu=1.1):
  rebalance_logits = logits * mu
  probs = softmax(rebalance_logits)
  pixel_dict = np.arange(0, 256, dtype=np.float32)
  pixels = np.sum(probs * pixel_dict, axis=1)
  return np.floor(pixels)