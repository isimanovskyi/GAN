import torch
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
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        flags.g_parser.add_argument('--'+name, default=default, type=str2bool, help=help)

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

def get_torch_device():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        cur_device = torch.cuda.current_device()
        device = torch.device('cuda:' + str(cur_device))
        logger.info('CUDA device: ' + torch.cuda.get_device_name(cur_device))
    else:
        device = torch.device('cpu:0')
        logger.info('CUDA not available')
    return device

def image_to_tensorboard(img):
    """Add image data to summary.
        Args:
            img (torch.Tensor): Image data
        Shape:
            img: :math:`(3, H, W)`. Expect it in range -1, 1
    """
    img = 127.5 * (img + 1.)
    return img.type(torch.ByteTensor)
