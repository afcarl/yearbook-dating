import sys
sys.path.insert(0,'../python')
import caffe

import numpy as np
import PIL
from PIL import Image
import scipy.io

import random

class YearbookDataLayer(caffe.Layer):
    """
    Load images from the Yearbook dataset in a batch, 
    feed to convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - data_dir: path to yearbook images dir
        - split: train / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        - mean: per-channel mean of dataset
        - im_shape: H x W dimensions of images (all are same size)
        - batch_size: size of SGD mini-batch
        - num_classes: N-way classfication problem

        example: params = dict(im_shape=(96,96), num_classes=83, batch_size=64, split=['train'], mean=(128, 128, 128), seed=1337)
        """
        # config
        params = eval(self.param_str)
        self.data_dir = params['yearbook_dir']
        self.split = params['split']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.mean = np.array(params['mean'])
        self.im_shape = params['im_shape'] 
        self.batch_size = int(params['batch_size'])
        self.num_classes = params.get('num_classes', 83)
        self.bad_imgs = []

        # tops: check configuration
        if len(top) != 2:
            raise Exception("Need to define 2 tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # Shape the data layer once since all the data is the same size
        top[0].reshape(self.batch_size,3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, 1) # label == scalar

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.data_dir, self.split[0])
        self.img_ids = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            random.shuffle(self.img_ids)

    def reshape(self, bottom, top):
        # No need since we already did reshape in setup
        # and all images are the same size!
        pass

    def load_next_image(self):
        if self.idx == len(self.img_ids):
            self.idx = 0
            if self.random:
                random.shuffle(self.img_ids)
        data = None
        while data is None:
            try:
                data = self.load_image(self.img_ids[self.idx])
                label = self.load_label(self.img_ids[self.idx])
            except:
                self.idx += 1
        self.idx += 1
        return data, label

    def forward(self, bottom, top):
	# assign output
        for itt in range(self.batch_size):
            data, label = self.load_next_image()
            top[0].data[itt, ...] = data
            top[1].data[itt, ...] = label
        
    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe
        """
        im = Image.open('{}/images/{}'.format(self.data_dir, idx))
        im = im.resize((self.im_shape), PIL.Image.BILINEAR)
        in_ = np.array(im, dtype=np.float32) 
        # NOTE: other data augmentations were tried to no effect
        # - random cropping from a padded image
        # - global contrast normalization
        if 'train' in self.split:
            # do a simple horizontal flip 50% of the time
            flip = np.random.choice(2)*2-1
            in_ = in_[:, ::flip, :]
        # Caffe pre-processing
        in_ = in_[:,:,::-1] # swap channel order RGB -> BGR
        in_ -= np.array(self.mean) # subtract mean
        in_ = in_.transpose((2,0,1)) # transpose to C x H x W
        return in_

    def load_label(self, idx):
        """
	Compute the label from the filename, which has format:
        year_state_city_high-school_ID.png
        We classify images between the years 1928 and 2010
        so labels are assigned continguously starting from 1928
        """
        min_year = 28
        year = idx.split('_')[0]
        if year[0] == str(1): # 20th century
            label = int(year[-2:]) - min_year 
        else: # 21st century
            label = int(year[-2:]) + (100 - min_year)
        return label

