"""Create image-list file
Example:
python tools/create_img_lists.py --dataset=data/celebA --outfile=data/train.txt
"""
import os
from optparse import OptionParser
from random import shuffle

# length=32

parser = OptionParser()
parser.add_option("--dataset", dest="dataset",  
                  help="dataset path")

parser.add_option("--outfile", dest="outfile",  
                  help="outfile path")
(options, args) = parser.parse_args()

f = open(options.outfile, 'w')
dataset_basepath = options.dataset

path=os.listdir(dataset_basepath)
path.sort(key=lambda f: int(filter(str.isdigit, f)))
# sh=path[length:]
# shuffle(sh)
# path[length:]=sh

i = 1
maximum = 207872
for p1 in path:
  if i > maximum:
    break
  image = os.path.abspath(dataset_basepath + '/' + p1)
  f.write(image + '\n')
  i+=1
f.close()
