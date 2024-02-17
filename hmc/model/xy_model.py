import numpy as np
from .spin_model import SpinModel
import torch



class XYModel(SpinModel):
    
    def __init__(self, grid_state, J=1, h=0, beta=1):   
        super().__init__(grid_state, J, h, beta)
    
    # theta range from -pi to pi
    def confine_theta(theta):
        return np.mod(theta, np.pi*2) - np.pi

    def _init_spins(self):
        # range [0, 2 pi]
        theta = np.random.randn(self.grid.n_sites) * 2 * np.pi
        # range [-pi, pi]
        theta = XYModel.confine_theta(theta)
        self.set_state(theta)
    
    # scalar magnetization
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
        
        energy_x = -self.J * np.dot(sx, self.K @ sx)/2
        energy_y = -self.J * np.dot(sy, self.K @ sy)/2
        # Without losing generality, suppose the external field is along x-axis
        energy_field = -self.h * np.sum(sx)
        energy = energy_x + energy_y + energy_field
        return energy
    
    # in HMC, the energy is treated as potential energy. 
    # The virtue kinetic energy comes from virtue momentum
    def get_potential(self, theta):
        potential = self.get_energy(theta) * self.beta
        return potential
    
    # for hamiltonian Monte Carlo only, momentum is virtue
    def get_hamiltonian(self, theta, momentum):
        kinetic = 0.5 * np.dot(momentum,momentum)
        potential = self.get_potential(theta)
        hamiltonian = kinetic + potential
        return hamiltonian
    
     # A torch version of hamiltonian function
    # for PyTorch autograd 
    def get_hamiltonian_torch(self, theta, momentum):

        K = torch.tensor(self.K, dtype=np.float)
        h = torch.tensor(self.h, dtype=np.float)
        J = torch.tensor(self.J, dtype=np.float)

        kinetic = 0.5* torch.dot(momentum, momentum)

        sx = torch.cos(theta)
        sy = torch.sin(theta)
        spin_interaction_x = torch.dot(sx, K @ sx) * (-J/2)
        spin_interaction_y = torch.dot(sy, K @ sy) * (-J/2)
        external_field = torch.sum(sx) * (-h)
        potential = spin_interaction_x +  spin_interaction_y + external_field
        potential = potential*self.beta
        hamiltonian = kinetic + potential
        return hamiltonian


    def get_gradU(self, theta):

        cos = np.cos(theta)
        sin = np.sin(theta)

        grad_U_x = -self.J * (-sin) * (self.K @ cos) 
        grad_U_y = -self.J * cos * (self.K @ sin)
        grad_U_field = -self.h * (-sin)

        grad_U =  grad_U_x + grad_U_y + grad_U_field
        return grad_U*self.beta
    
    

    