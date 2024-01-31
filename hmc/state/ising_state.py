import numpy as np
from scipy.sparse import coo_matrix
from .grid_state import GridState
import torch

class IsingState:
    
    def __init__(self, grid_state, J=1, h=0, beta=1):

        self.grid = grid_state
        self._init_spins()
        
        self.J = J # coupling between spins
        self.h = h #  external field
        self.beta=beta # inverse temperature
        self.K = self.grid.get_adjacent_matrix() # adjacent matrix
    
    def _init_spins(self):
        # generate sequence of [0,1,0,1,0,0...]
        self.grid.sites = np.random.randint(2, size=self.grid.n_sites)
        # convert [0,1,0,1...] to [-1,1,-1,1]...
        self.sites = self.sites * 2 -1
    
    def get_magnetization(self, spins):
        magnetization= np.mean(spins)
        return magnetization
    
    def get_energy(self, spins):
        energy = 0
        # interaction energy between spins
        energy += -self.J * np.dot(spins, self.K @ spins)/2
        # energy of external field
        energy += -self.h * np.sum(spins)
        return energy
    
