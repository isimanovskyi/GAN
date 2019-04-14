import numpy as np
import logger

class Constant(object):
    def __init__(self, lambd):
        self.lambd = lambd

    def update(self, errD):
        return False

    def __float__(self):
        return self.lambd

    def get(self):
        return self.lambd

    def set(self, value):
        if not isinstance(value, float):
            raise ValueError('should be float')

        #self.lambd = value

class ThresholdAnnealing(object):
    def __init__(self, lambd, beta=0.99, threshold=1.1, min_switch_step=1000, verbose=False):
        self.verbose = verbose
        self.lambd = lambd
        self.beta = beta
        self.threshold = threshold
        self.min_switch_step = min_switch_step
        
        self.errD_av = 0.
        self.step = 0
        self.step_hit = 0
        self.step_switched = 0

    def get_average(self):
        t = max(float(self.step), 1.)
        return self.errD_av / (1. - np.power(self.beta, t))

    def switch(self):
        if self.step < self.step_switched + 100:
            return False

        self.lambd /= 2.
        self.step_switched = self.step
        logger.event('[%d] lambda switched: from %4.4f to %4.4f; %.8f' % (self.step, self.lambd * 2, self.lambd, self.get_average()))
        return True

    def update(self, errD):
        prev_value = self.get_average()

        self.step += 1
        self.errD_av = self.beta * self.errD_av + (1.-self.beta)*errD

        bSwitch = False
        if self.get_average() > self.threshold:
            if self.step > self.step_hit + self.min_switch_step:
                bSwitch = self.switch()
                if bSwitch:
                    self.step_hit = self.step

            if prev_value < self.threshold:
                self.step_hit = self.step

        if self.verbose:
            logger.info("lambda: %.8f, %d, %d" % (self.get_average(), self.step_hit, self.step_switched))
        return bSwitch

    def __float__(self):
        return self.lambd

    def get(self):
        return self.lambd

    def set(self, value):
        if not isinstance(value, float):
            raise ValueError('should be float')

        self.lambd = value