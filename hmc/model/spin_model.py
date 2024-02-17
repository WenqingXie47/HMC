import numpy as np


class SpinModel:

    def __init__(self, grid_state, J=1, h=0, beta=1):

        self.grid = grid_state
        self._init_spins()
        
        self.J = J # coupling between spins
        self.h = h #  external field
        self.beta=beta # inverse temperature
        self.K = self.grid.get_adjacent_matrix() # adjacent matrix
    

    def get_state(self):
        return self.grid.get_state()

    def set_state(self, new_state):
        self.grid.set_state(new_state)


    def _init_spins(self):
        pass
    
    def get_magnetization(self, spins):
        pass
    
    def get_energy(self, spins):
        pass
    