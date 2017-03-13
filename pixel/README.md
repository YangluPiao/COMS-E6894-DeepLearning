# Pixel Recursive Super Resolution

TensorFlow implementation of [Pixel Recursive Super Resolution](https://arxiv.org/abs/1702.00783). 

## Requirements

- Python 2.7
- [Skimage](http://scikit-image.org/)
- [TensorFlow](https://www.tensorflow.org/) 1.0


## Usage

First, download data [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

    $ mkdir data
	$ cd data
	$ ln -s $celebA_path celebA

Then, create image_list file:

	$ python /create_img_lists.py --dataset=data/celebA --outfile=data/train.txt

To train model on gpu:

	$ python /train.py
	(or $ python tools/train.py --device_id=0)

To train model on cpu:
	$ python /train.py --use_gpu=False

## Author

nilboy / [@nilboy](https://github.com/nilboy)