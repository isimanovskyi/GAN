import torch
import torchvision

import PIL
from glob import glob

class ImageFiles(torch.utils.data.Dataset):
    def __init__(self, path, grayscale = False, output_shapes = None, center_crop = None):
        #base transforms + augmentation
        transforms_list = []
        if grayscale:
            transforms_list.append(torchvision.transforms.Grayscale())
        if center_crop is not None:
            transforms_list.append(torchvision.transforms.CenterCrop(center_crop))

        #apply different output shapes
        out_tr = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]
        if output_shapes is None:
            transforms_list += out_tr
            self.transforms = torchvision.transforms.Compose(transforms_list)
            self.single_transform = True
        else:
            try:
                self.transforms = []
                for sh in output_shapes:
                    t_lst = [torchvision.transforms.Resize(sh)] + out_tr
                    transform = torchvision.transforms.Compose(transforms_list + t_lst)
                    self.transforms.append(transform)

                if len(self.transforms) == 0:
                    transforms_list += out_tr
                    self.transforms = torchvision.transforms.Compose(transforms_list)
                    self.single_transform = True

                elif len(self.transforms) == 1:
                    self.transforms = self.transforms[0]
                    self.single_transform = True

                else:
                    self.single_transform = False

            except TypeError:
                transforms_list += out_tr
                self.transforms = torchvision.transforms.Compose(transforms_list)
                self.single_transform = True

        #split data to train and test
        self.data_files = glob(path)
        if len(self.data_files) == 0:
            raise ValueError('no files')

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        if type(item) == torch.Tensor:
            item = item.item()

        img = PIL.Image.open(self.data_files[item])
        if self.single_transform:
            return self.transforms(img)

        return [t(img) for t in self.transforms]

