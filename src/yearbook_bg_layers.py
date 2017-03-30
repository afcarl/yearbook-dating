import caffe
import numpy as np
import PIL
from PIL import Image
import scipy.io
import random

class YearbookDataLayer(caffe.Layer):
    """
    Load random background crops from the Yearbook dataset
    """

    def setup(self, bottom, top):
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

        # tops: check configuration
        if len(top) != 2:
            raise Exception("Need to define 2 tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # Shape the data layer once since all the data is the same size
        top[0].reshape(self.batch_size,3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, 1) 

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
        pass

    def load_next_image(self):
        if self.idx == len(self.img_ids):
            self.idx = 0
            if self.random:
                random.shuffle(self.img_ids)
        data = self.load_image(self.img_ids[self.idx])
        label = self.load_label(self.img_ids[self.idx])
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
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = np.array(Image.open('{}/images/{}'.format(self.data_dir, idx)), dtype=np.uint8)
        # Take a random 32 x 32 crop out of one of the four corners of the image
        corner_points = [(0,0), (im.shape[0] -32, 0), (0, im.shape[1] -32), (im.shape[0] -32, im.shape[1]-32)]
        pt = random.choice(corner_points)
        crop = im[pt[0]: pt[0] + 32, pt[1]:pt[1] + 32]
        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = crop[:, ::flip, :]
        # Caffe pre-processing
        in_ = im.astype(np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        """
	Compute the label from the filename
        """
        min_year = 28
        year = idx.split('_')[0]
        if year[0] == str(1): # 20th century
            label = int(year[-2:]) - min_year 
        else: # 21st century
            label = int(year[-2:]) + (100 - min_year)
        return label

    
