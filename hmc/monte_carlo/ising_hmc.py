
from .monte_carlo import MonteCarlo
import numpy as np



class IsingHMC(MonteCarlo):

    def __init__(self, ising_state, integrator, autograd=False):
        
        self.state = ising_state
        self.integrator = integrator
        self.autograd = autograd
        if self.autograd :
            self.generator_func = self.state.get_hamiltonian_torch
        else:
            self.generator_func = self.state.get_gradU



    def update(self, psi):
        
        # randomize a virtue momentum
        momentum = np.random.randn(self.state.n_sites)
        # evolve in phase space according to hamiltonian
        new_psi, new_momentum = self.integrator.update(self.generator_func,psi,momentum)
        # compare old and new hamiltonians and decide to accept new state or not
        hamiltonian = self.state.get_hamiltonian(psi,momentum)
        new_hamiltonian = self.state.get_hamiltonian(new_psi,new_momentum)
        # print(hamiltonian-new_hamiltonian)
        if np.random.rand() < np.exp(hamiltonian-new_hamiltonian):
            return new_psi
        else:
            return psi
        





    
