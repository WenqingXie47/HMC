import numpy as np

class XYState:
    
    def __init__(self, grid_state, J=1, h=0, beta=1):
        
        self.grid = grid_state
        self._init_spins()
        
        self.J = J # coupling between spins
        self.h= h #  external field
        self.beta=beta # inverse temperature
        self.K = self.grid.get_adjacent_matrix() # adjacent matrix
    
    def _init_spins(self):
        # range from [0, 2 pi]
        self.grid.sites = np.random.randn(self.grid.n_sites) * 2 * np.pi
       
    
    def get_magnetization(self, theta):
        
        sx = np.cos(theta)
        sy = np.sin(theta)
        
        mx = np.mean(sx)
        my = np.mean(sy) 
        m = np.sqrt(mx**2 + my**2)
        return m
    
    def get_energy(self, theta):
        
        sx = np.cos(theta)
        sy = np.sin(theta)
        
        energy = 0
        energy += -self.J * np.dot(sx, self.K @ sx)/2
        energy += -self.J * np.dot(sy, self.K @ sy)/2
        # Without losing generality, suppose the external field is along x-axis
        energy += -self.h * np.sum(sx)
        return energy