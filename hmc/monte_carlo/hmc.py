
from .monte_carlo import MonteCarlo
import numpy as np



class IsingHMC(MonteCarlo):

    def __init__(self, ising_model, integrator, autograd=False):
        
        self.model = ising_model
        self.initial_state = ising_model.get_state()

        self.integrator = integrator
        self.autograd = autograd

        if self.autograd :
            self.generator_func = self.model.get_hamiltonian_torch
        else:
            self.generator_func = self.model.get_gradU
        self.get_potential = self.model.get_potential

    
    def get_kinetic(mom):
        kinetic = 0.5 * np.dot(mom,mom)
        return kinetic

    def get_hamiltonian(self, psi, mom):
        kinetic = IsingHMC.get_kinetic(mom)
        potential = self.get_potential(psi)
        hamiltonian = kinetic+potential
        return hamiltonian


    def update(self, psi):
        
        # randomize a virtue momentum
        momentum = np.random.randn(self.model.grid.n_sites)
        # evolve in phase space according to hamiltonian
        new_psi, new_momentum = self.integrator.update(self.generator_func,psi,momentum)
        # compare old and new hamiltonians and decide to accept new state or not
        hamiltonian = self.get_hamiltonian(psi,momentum)
        new_hamiltonian = self.get_hamiltonian(new_psi,new_momentum)

        delta_hamiltonian = new_hamiltonian-hamiltonian
        if np.random.rand() < np.exp(-delta_hamiltonian):
            return new_psi
        else:
            return psi
        





class XYHMC(IsingHMC):

    
    def update(self, psi):
        
        new_psi = super().update(psi)
        new_psi = self.model.confine_theta(new_psi)
        return new_psi
       