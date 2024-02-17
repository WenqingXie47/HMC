import numpy as np
from .spin_model import SpinModel

class IsingModel(SpinModel):
    
    def __init__(self, grid_state, J=1, h=0, beta=1):
        super().__init__(grid_state, J, h, beta)
        
        
    def _init_spins(self):
        # generate sequence of [0,1,0,1,0,0...]
        spins = np.random.randint(2, size=self.grid.n_sites)
        # convert [0,1,0,1...] to [-1,1,-1,1]...
        spins = self.sites * 2 -1
        self.set_state(spins)
    

    def get_magnetization(self, spins):
        magnetization= np.mean(spins)
        return magnetization
    

    def get_energy(self, spins):    
        # interaction energy between spins
        spin_interaction_energy = -self.J * np.dot(spins, self.K @ spins)/2
        # energy of external field
        external_field_energy = -self.h * np.sum(spins)
        energy = spin_interaction_energy + external_field_energy
        return energy
    

    
    

    
