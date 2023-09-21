import torch

class LeapFrogIntegrator:

    def __init__(self, step_size=0.1, n_steps=10):
        self.step_size = step_size
        self.n_steps = n_steps

    def update(self, gradU_func, pos, mom):

        mom = mom - 0.5 * self.step_size * gradU_func(pos)
        for i in range(self.n_steps):
            # Make a full step for the position
            pos = pos +  self.step_size * mom
            # Make a full step for the momentum, except at end of trajectory
            if (i<self.n_steps-1):
                mom = mom - self.step_size * gradU_func(pos)
        mom = mom - 0.5 * self.step_size * gradU_func(pos)
        return pos, mom


