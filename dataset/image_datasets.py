import os
import dataset.image_files

class CelebA(dataset.image_files.ImageFiles):
    name = 'celeba'
    path = os.path.join('./data', name)

    def __init__(self, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        super(CelebA, self).__init__(os.path.join(CelebA.path, '*.jpg'), batch_size, grayscale, output_shapes, center_crop)

class ImageNet(dataset.image_files.ImageFiles):
    name = 'imagenet'
    path = os.path.join('./data', name)

    def __init__(self, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        super(ImageNet, self).__init__(os.path.join(ImageNet.path, "*.png"), batch_size, grayscale, output_shapes, center_crop)

class LSunBedroom(dataset.image_files.ImageFiles):
    name = 'lsun_bedroom'
    path = "./data/lsun/bedroom"

    def __init__(self, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        super(LSunBedroom, self).__init__(os.path.join(LSunBedroom.path, "*.jpg"), batch_size, grayscale, output_shapes, center_crop)

