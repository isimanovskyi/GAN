import os
import dataset.image_files

class CelebA(dataset.image_files.ImageFiles):
    def __init__(self, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        super(CelebA, self).__init__(os.path.join("./data", 'celebA', "*.jpg"), batch_size, grayscale, output_shapes, center_crop)

class ImageNet(dataset.image_files.ImageFiles):
    def __init__(self, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        super(ImageNet, self).__init__(os.path.join("./data", 'imageNet', "*.png"), batch_size, grayscale, output_shapes, center_crop)

class LSunBedroom(dataset.image_files.ImageFiles):
    def __init__(self, batch_size, grayscale = False, output_shapes = None, center_crop = None):
        super(LSunBedroom, self).__init__(os.path.join("./data", 'lsun/bedroom', "*.jpg"), batch_size, grayscale, output_shapes, center_crop)