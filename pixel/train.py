from __future__ import print_function
import tensorflow as tf
from solver import *

flags = tf.app.flags

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
#solver
data_file="data/train.txt"
flags.DEFINE_string("train_dir", "models", "trained model save path")
flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
flags.DEFINE_string("imgs_list_path", data_file , "images list file path")

flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")
flags.DEFINE_integer("device_id", 0, "gpu device id")

flags.DEFINE_integer("num_epoch", 15, "train epoch num")

flags.DEFINE_integer("batch_size", 32, "batch_size")
print("size of batch:",flags.FLAGS.batch_size)

s=file_len(data_file)
flags.DEFINE_integer("file_length", s, "file_length")
print("number of images for training: ",s)
# flags.DEFINE_integer("dataset_size", s, "size of dataset")

flags.DEFINE_integer("size_hr", 32, "size of high resolution images")
flags.DEFINE_integer("size_lr", 8, "size of low resolution image")

flags.DEFINE_float("learning_rate", 4e-4, "learning rate")


conf = flags.FLAGS

def main(_):
  solver = Solver()
  solver.train()

if __name__ == '__main__':
  tf.app.run()