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
flags.DEFINE_string("train_dir", "models", "trained model save path")
flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
flags.DEFINE_string("imgs_list_path", "data/train.txt", "images list file path")

flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")
flags.DEFINE_integer("device_id", 0, "gpu device id")

flags.DEFINE_integer("num_epoch", 30, "train epoch num")
print("size of batch:",32)
flags.DEFINE_integer("batch_size", 32, "batch_size")

s=file_len("data/train.txt")
print("number of samples:",s)
flags.DEFINE_integer("dataset_size", s, "size of dataset")

flags.DEFINE_float("learning_rate", 4e-4, "learning rate")


conf = flags.FLAGS

def main(_):
  solver = Solver()
  solver.train()

if __name__ == '__main__':
  tf.app.run()