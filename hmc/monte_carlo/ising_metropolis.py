from ..integrator.leap_frog import LeapFrogIntegrator
import numpy as np
import torch

class IsingMetropolis:

    def __init__(self, ising_state):
        self.state = ising_state
       

    def update(self, psi):
        
        # evolve in phase space according to hamiltonian
        new_psi = psi + np.random.randn(self.state.n_sites)
        # compare old and new hamiltonians and decide to accept new state or not
        hamiltonian = self.state.get_potential(psi)
        new_hamiltonian = self.state.get_potential(new_psi)
        if np.random.rand() < np.exp(hamiltonian-new_hamiltonian):
            return new_psi
        else:
            return psi


    def thermalize(self, psi, n_iters=10000):
        for _ in range(n_iters):
            psi = self.update(psi)
        return psi 
    

    def sample(self, measurements, n_samples=500, n_iters_per_sample=20, n_thermalization_iters=10000):
        # measurements is a dict store the functions of psi
        # e.g.: {"magnetization": get_magnetization, "energy", get_energy}
        # where get_magnetization(psi) = m
        
        # create a dictionary to record measurement history
        result_dict = {}
        for key in measurements.keys():
            result_dict[key] = []

        psi = self.state.get_state_vector()
        psi = self.thermalize(psi,n_thermalization_iters)
        for _ in range(n_samples):

            for key, measurement_func in measurements.items():
                result_dict[key].append(measurement_func(psi))
            
            for _ in range(n_iters_per_sample):
                psi = self.update(psi)

        # turn results into np.ndarray
        for key in measurements.keys():
            result_dict[key] = np.array(result_dict[key])

        return result_dict