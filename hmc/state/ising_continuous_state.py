import numpy as np
import torch

class IsingContinuousState:

    def __init__(self, grid_state, J=1, h=0, beta=1):

        
        self.grid = grid_state
        self._init_spins()
        
        self.J = J # coupling between spins
        self.h= h*np.ones(shape=(self.grid.n_sites),dtype=float) # external field
        self.beta=beta # inverse temperature
        self.K = self.grid.get_adjacent_matrix() # adjacent matrix
        self._init_effective_parameters()


    def _init_spins(self):
        self.grid.sites = np.random.randn(self.grid.n_sites)

    def _init_effective_parameters(self):
        # constant C ensures K+CI is positive definite
        self.C = self.grid.n_neighbours + 1e-3 
        self.K_prime = self.K + self.C * np.identity(self.grid.n_sites, dtype=np.float)
        self.J_prime = self.beta * self.J
        self.h_prime = self.beta * self.h 


    def get_average(self,psi):
        return np.mean(psi)


    def get_magnetization(self, psi):

        magnetization= np.mean(np.tanh(self.J_prime * self.K_prime @ psi + self.h_prime))
        return magnetization
    

    def get_potential(self, psi):
        spin_interaction = 0.5 * self.J_prime * np.dot(psi, self.K_prime @ psi)
        log_cosh  = -np.sum(np.log(np.cosh(self.J_prime*(self.K_prime @ psi)+self.h_prime)))
        potential = spin_interaction + log_cosh
        return potential
    
    def get_gradU(self, psi):

        grad_U =  self.J_prime * self.K_prime @ (psi - np.tanh(self.J_prime * self.K_prime @ psi + self.h_prime)) 
        return grad_U
    

    def get_hamiltonian(self, psi, momentum):
        kinetic = 0.5 * np.dot(momentum,momentum)
        potential = self.get_potential(psi)
        hamiltonian = kinetic + potential
        return hamiltonian
    

    # A torch version of hamiltonian function
    # for PyTorch autograd 
    def get_hamiltonian_torch(self, psi, momentum):
        K_prime = torch.tensor(self.K_prime)
        h_prime = torch.tensor(self.h_prime)
        J_prime = torch.tensor(self.J_prime)

        kinetic = 0.5* torch.dot(momentum, momentum)
        spin_interaction = 0.5 * J_prime * torch.dot(psi, K_prime @ psi)
        log_cosh  = -torch.sum(torch.log(torch.cosh(J_prime*(K_prime @ psi) + h_prime)))
        potential = spin_interaction +  log_cosh
        hamiltonian = kinetic + potential
        return hamiltonian
        