import numpy as np

class MonteCarlo:

    def __init__(self, initial_state):
        self.initial_state = initial_state
       

    def update(self, psi):
        pass


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

        psi = self.initial_state
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