import os
import dataset.image_files

class CelebA(dataset.image_files.ImageFiles):
    name = 'celeba'

    def __init__(self, data_folder, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        path = os.path.join(data_folder, CelebA.name, '*.jpg')
        super(CelebA, self).__init__(path, grayscale, output_shapes, center_crop)

class ImageNet(dataset.image_files.ImageFiles):
    name = 'imagenet'

    def __init__(self, data_folder, grayscale = False, output_shapes = None, center_crop = None):
        path = os.path.join(data_folder, ImageNet.name, '*.png')
        super(ImageNet, self).__init__(path, grayscale, output_shapes, center_crop)

class LSunBedroom(dataset.image_files.ImageFiles):
    name = 'lsun_bedroom'

    def __init__(self, data_folder, grayscale = False, output_shapes = None, center_crop = None):
        path = os.path.join(data_folder, 'lsun/bedroom', '*.jpg')
        super(LSunBedroom, self).__init__(path, grayscale, output_shapes, center_crop)

