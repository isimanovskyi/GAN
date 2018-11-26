import numpy as np
from multiprocessing import Queue
import threading

class ThreadedBatch(object):
    def __init__(self, batch, queue_size = 10):
        self.batch = batch
        self.queue = Queue(queue_size)
        
        self.thread = threading.Thread(target=self.thread_proc)
        self.thread.start()

    def thread_proc(self):
        while True:
            self.queue.put(self.batch.get_batch())

    def get_size(self):
        return self.batch.get_size()

    def get_samples(self, size):
        return self.batch.get_samples(size)

    def get_z(self):
        return self.batch.get_z()

    def get_x(self):
        raise NotImplementedError('Not implemented')

    def get_batch(self):
        return self.queue.get()

    def get_z_shape(self):
        return self.batch.get_z_shape()

    def get_image_shape(self):
        return self.batch.get_image_shape()


class BatchWithNoise(object):
    def __init__(self, dataset, z_dim = 100):
        self.z_shape = (z_dim,)
        self.dataset = dataset

    def get_size(self):
        return self.dataset.batch_size

    def get_samples(self, size):
        sample_images = self.dataset.get_samples(size)[0]
        sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(size,) + self.z_shape).astype(np.float32)
        return sample_seed, sample_images

    def get_z(self):
        return np.random.normal(loc=0.0, scale=1.0, size=(self.dataset.batch_size,) + self.z_shape).astype(np.float32)

    def get_x(self):
        return self.dataset.get_batch()[0]

    def get_batch(self):
        return self.get_z(), self.get_x()

    def get_z_shape(self):
        return self.z_shape

    def get_image_shape(self):
        return (3,) + self.dataset.output_shapes[0]

class BatchSRez(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def get_samples(self, size):
        sample_images = self.dataset.get_samples(size)
        return sample_images[0], sample_images[1]

    def get_z(self):
        raise NotImplementedError('Not implemented')

    def get_x(self):
        raise NotImplementedError('Not implemented')

    def get_batch(self):
        batch_x = self.dataset.get_batch()
        if batch_x is None:
            return None, None

        return batch_x[0], batch_x[1]

    def get_z_shape(self):
        return (3,) + self.dataset.output_shapes[0]

    def get_image_shape(self):
        return (3,) + self.dataset.output_shapes[1]