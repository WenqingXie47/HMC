from hmc.state.grid_state import GridState
from hmc.model.xy_model import XYModel
from hmc.monte_carlo.hmc import XYHMC
from hmc.integrator.leap_frog import LeapFrogIntegrator


import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt


def measure(beta_list):
    energy_list = []
    magnetization_list=[]
    m2_list =[]
    heat_capacity_list=[]

    for beta in beta_list:
        model = XYModel(grid_state=grid,beta=beta, J=J, h=h) # define ising state
        mc = XYHMC(ising_model=model,integrator=leap_frog) # define Monte Carlo algorithm
        measurements = {"magnetization": model.get_magnetization, "energy":model.get_energy} # define physics observable to be measured
        history = mc.sample(measurements=measurements,n_samples=4000,n_iters_per_sample=100,n_thermalization_iters=4000)
        
        energy_list.append(np.mean(history["energy"])/model.grid.n_sites)
        magnetization_list.append(np.mean(history["magnetization"]))
        m2_list.append(np.mean(np.square(history["magnetization"])))
        heat_capacity_list.append(np.var(history["energy"])*beta*beta/model.grid.n_sites)
   
        measurement = {
            "beta": beta_list, 
            "magnetization": magnetization_list,
            "m2": m2_list,
            "energy": energy_list,
            "heat_capacity": heat_capacity_list,
        }
        return measurement



if __name__ == "__main__":

    dim = 2 # dimension of Ising model
    L = 16 # length of grid
    grid = GridState(dim=2,length=6)

    J = 1.0 # spin-spin interaction coupling
    h = 0.0  # External field

    step_size = 0.1 # step size of integrator
    trajectory_length = 1
    n_steps = int(trajectory_length/step_size) # number of steps in a trajectory
    leap_frog = LeapFrogIntegrator(step_size, n_steps) # define integrator

    beta_list1 = np.linspace(0.5,1.5,10+1) # define beta valueus to measure
    beta_list2 = np.linspace(0.7,1.0, 15+1)
    beta_list = np.concatenate((beta_list1,beta_list2))
    beta_list = np.unique(np.sort(beta_list))

    measurement  = measure(beta_list)
    df = pd.DataFrame(data=measurement)
    filename = "./data/xy_model.csv"
    df.to_csv(filename)
        