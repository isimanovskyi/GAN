from random import shuffle
import scipy.misc
import numpy as np
from glob import glob
import os
import utils

#ignoring scipy deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def normalize(img):
    img = np.array(img).astype(np.float32)
    img = img/127.5 - 1.
    return img

class ImageFiles(object):
    def __init__(self, path, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        self.grayscale = grayscale
        self.output_shapes = output_shapes
        self.center_crop = center_crop
        self.batch_size = batch_size

        #split data to train and test        
        self.data_files = glob(path)
        if len(self.data_files) == 0:
            raise ValueError('no files')

        self.valid_files = self.data_files[0:300]
        self.data_files = self.data_files[300:]
        
        self.n_batch = 0
        self.n_batches = len(self.data_files) // batch_size

    def get_batch(self):
        if self.n_batch >= self.n_batches:
            self.n_batch = 0
            shuffle(self.data_files)

        batch_files = self.data_files[self.n_batch * self.batch_size : (self.n_batch+1) * self.batch_size]
        self.n_batch += 1

        return self.read_files(batch_files)

    def get_samples(self, size, index = None, s = 'valid'):
        if index is None:
            index = slice(0, size)

        if s == 'valid':
            sample_files = self.valid_files[index]
        elif s == 'train':
            sample_files = self.data_files[index]
        else:
            raise ValueError('Unknown argument')
        return self.read_files(sample_files)

    def read_files(self, files):
        res = []
        for sh in self.output_shapes:
            res.append(np.array([self.get_image(file, sh) for file in files]))
        return res


    def get_image(self, path, output_shape):
        if (self.grayscale):
            img = scipy.misc.imread(path, flatten = True)
        else:
            img = scipy.misc.imread(path)

        if self.center_crop is not None:
            crop_w = self.center_crop[0]
            crop_h = self.center_crop[1]

            h, w = img.shape[:2]
            
            if w > crop_w:
                x = int(round((w - crop_w)/2.))
            else:
                x = 0
                crop_w = w

            if h > crop_h:
                y = int(round((h - crop_h)/2.))
            else:
                y = 0
                crop_h = h
            
            img = img[y:y+crop_h, x:x+crop_w]

        if output_shape is not None:
            img = normalize(scipy.misc.imresize(img, output_shape))
        else:
            img = normalize(img)
        img = np.rollaxis(img,2,0)
        return img