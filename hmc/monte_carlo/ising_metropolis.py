from ..integrator.leap_frog import LeapFrogIntegrator
from . monte_carlo import MonteCarlo
import numpy as np




class IsingMetropolis(MonteCarlo):

    def __init__(self, ising_state):
        self.state = ising_state
        self.initial_state = ising_state.grid.get_state()
       

    def update(self, psi):
        
        # evolve in phase space according to hamiltonian
        new_psi = psi + np.random.randn(self.state.grid.n_sites)
        # compare old and new hamiltonians and decide to accept new state or not
        hamiltonian = self.state.get_potential(psi)
        new_hamiltonian = self.state.get_potential(new_psi)
        if np.random.rand() < np.exp(hamiltonian-new_hamiltonian):
            return new_psi
        else:
            return psi


 