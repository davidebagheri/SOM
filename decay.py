import numpy as np


class ExpDecay:
    def __init__(self, init_value, decay_step, decay_rate):
        self.init_value = init_value
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.last_value = init_value

    def get_value(self, epoch):
        decayed_value = self.last_value * np.power(self.decay_rate, (epoch / self.decay_step))
        self.last_value = decayed_value

        return decayed_value
