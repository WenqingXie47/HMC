from .monte_carlo import MonteCarlo
import numpy as np




class IsingContinuousMetropolis(MonteCarlo):

    def __init__(self, ising_model):
        self.model = ising_model
        self.initial_state = ising_model.grid.get_state()
       
    def get_delta_energy(self, psi, new_psi):
        energy = self.model.get_energy(psi)
        new_energy = self.model.get_energy(new_psi)
        delta_energy = new_energy-energy
        return delta_energy

    def update(self, psi):
        
        # evolve in phase space according to hamiltonian
        new_psi = psi + np.random.randn(self.model.grid.n_sites)
        # compare old and new hamiltonians and decide to accept new state or not
        delta_energy = self.get_delta_energy(psi, new_psi)
        beta = self.model.beta
        if np.random.rand() < np.exp(-delta_energy*beta):
            return new_psi
        else:
            return psi
        

# if we have a generator, which can generate some similar distribution
class FlowXYMetropolis(MonteCarlo):

    def __init__(self, ising_model, state_generator):
        self.model = ising_model
        self.initial_state = ising_model.grid.get_state()
        self.state_generator = state_generator
       
    def get_delta_energy(self, psi, new_psi):
        energy = self.model.get_energy(psi)
        new_energy = self.model.get_energy(new_psi)
        delta_energy = new_energy-energy
        return delta_energy


    def update(self, psi, log_prob):
        
        # generate new state
        new_psi, new_log_prob = self.state_generator.generate()
        # compare old and new hamiltonians and decide to accept new state or not
        delta_energy = self.get_delta_energy(psi, new_psi)
        delta_log_prob = new_log_prob - log_prob
        beta = self.model.beta
        if np.random.rand() < np.exp(-delta_energy*beta - delta_log_prob):
            return new_psi, new_log_prob
        else:
            return psi, log_prob
    
    def sample(self, measurements, n_samples=500, n_iters_per_sample=10, n_thermalization_iters=0):
        # measurements is a dict store the functions of psi
        # e.g.: {"magnetization": get_magnetization, "energy", get_energy}
        # where get_magnetization(psi) = m
        
        # create a dictionary to record measurement history
        result_dict = {}
        for key in measurements.keys():
            result_dict[key] = []

        psi = self.initial_state
        psi = self.thermalize(psi,n_thermalization_iters)
        for _ in range(n_samples):

            for key, measurement_func in measurements.items():
                result_dict[key].append(measurement_func(psi))
            
            for _ in range(n_iters_per_sample):
                psi, log_prob = self.update(psi, log_prob)

        # turn results into np.ndarray
        for key in measurements.keys():
            result_dict[key] = np.array(result_dict[key])

        return result_dict


 