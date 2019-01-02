import numpy as np
import logger

class Constant(object):
    def __init__(self, lambd):
        self.lambd = lambd

    def update(self, errD):
        pass

    def __float__(self):
        return self.lambd

    def get(self):
        return self.lambd

    def set(self, value):
        if not isinstance(value, float):
            raise ValueError('should be float')

        #self.lambd = value

class ThresholdAnnealing(object):
    def __init__(self, lambd, beta = 0.99, threshold = 1.1, min_switch_step = 1000):
        self.lambd = lambd
        self.beta = beta
        self.threshold = threshold
        self.min_switch_step = min_switch_step
        
        self.errD_av = 0.
        self.step = 0
        self.step_switched = 0

    def update(self, errD):
        self.step += 1
        t = float(self.step)

        self.errD_av = self.beta * self.errD_av + (1.-self.beta)*errD
        errD_av_corrected = self.errD_av/(1.-np.power(self.beta, t))

        if errD_av_corrected > self.threshold and self.step > self.step_switched + self.min_switch_step:
            self.step_switched = self.step
            self.lambd /= 2.
            logger.event('[%d] lambda switched: from %4.4f to %4.4f'% (self.step, self.lambd*2, self.lambd))

    def __float__(self):
        return self.lambd

    def get(self):
        return self.lambd

    def set(self, value):
        if not isinstance(value, float):
            raise ValueError('should be float')

        self.lambd = value