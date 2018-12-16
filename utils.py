import numpy as np
import os
import logger
import argparse
import imageio

class flags(object):
    g_parser = argparse.ArgumentParser()

    @staticmethod
    def DEFINE_string(name, default, help):
        flags.g_parser.add_argument('--'+name, default=default, type=str, help=help)

    @staticmethod
    def DEFINE_integer(name, default, help):
        flags.g_parser.add_argument('--'+name, default=default, type=int, help=help)

    @staticmethod
    def DEFINE_float(name, default, help):
        flags.g_parser.add_argument('--'+name, default=default, type=float, help=help)

    @staticmethod
    def DEFINE_boolean(name, default, help):
        flags.g_parser.add_argument('--'+name, default=default, type=bool, help=help)

    @staticmethod
    def FLAGS():
        args = flags.g_parser.parse_args()
        return args

def exists_or_mkdir(path, verbose=True):
    if not os.path.exists(path):
        if verbose:
            logger.info("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            logger.info("[!] %s exists ..." % path)
        return True

def save_images(images, size, image_path):
    if len(images.shape) == 3:  # Greyscale [batch, h, w] --> [batch, h, w, 1]
        images = images[:, :, :, np.newaxis]

    images = np.transpose(images,(0,2,3,1))

    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3), dtype=images.dtype)
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    def imsave(images, size, path):
        if np.max(images) <= 1 and (-1 <= np.min(images) < 0):
            images = ((images + 1) * 127.5).astype(np.uint8)
        elif np.max(images) <= 1 and np.min(images) >= 0:
            images = (images * 255).astype(np.uint8)

        return imageio.imwrite(path, merge(images, size))

    if len(images) > size[0] * size[1]:
        raise AssertionError("number of images %s should be equal or less than size[0] * size[1] %s" % (len(images), size[0] * size[1]))

    return imsave(images, size, image_path)

def is_str(val):
    t = type(val)
    
    if t is str:
        return True
    
    try:
        if t is unicode:
            return True
    except NameError:
        pass

    return False

def get_items(d):
    if hasattr(d, 'iteritems'):
        return d.iteritems()

    return d.items()
