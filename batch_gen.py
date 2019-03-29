import numpy as np
import torch

class BatchBase(object):
    def __init__(self, dataset, batch_size, num_workers):
        self.batch_size = batch_size
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers,drop_last = True)

    def get(self):
        if not hasattr(self, 'it'):
            self.it = iter(self.data_loader)

        while True:
            try:
                return next(self.it)
            except StopIteration:
                self.it = iter(self.data_loader)

class BatchWithNoise(BatchBase):
    def __init__(self, dataset, batch_size, z_shape, num_workers = 1):
        super(BatchWithNoise,self).__init__(dataset,batch_size,num_workers)

        self.z_shape = z_shape

    def get_z(self):
        return torch.randn(self.batch_size, *self.z_shape)

    def sample_z(self, n):
        return torch.randn(n, *self.z_shape)

    def get_images(self):
        return super(BatchWithNoise,self).get()

    def get(self):
        z = self.get_z()
        images = self.get_images()
        return {'z':z, 'images':images}
