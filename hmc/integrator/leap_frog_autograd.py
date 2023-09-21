from .leap_frog import LeapFrogIntegrator
import torch 

class LeapFrogIntegratorAutoGrad(LeapFrogIntegrator):

    def __init__(self, step_size=0.1, n_steps=10):
        super().__init__(step_size, n_steps)
        


    def get_grad(self, hamiltonian_func, pos, mom):
        energy = hamiltonian_func(pos,mom)
        energy.backward()
        pos_grad, mom_grad = pos.grad.data, mom.grad.data

        pos.grad = None
        mom.grad = None
        return pos_grad, mom_grad
        

    def update(self, hamiltonian_func, pos, mom):

        pos = torch.tensor(pos, requires_grad=True)
        mom = torch.tensor(mom, requires_grad=True)
    
        pos_grad, _ = self.get_grad(hamiltonian_func,pos,mom)
        mom.data -= 0.5 * self.step_size * pos_grad
        for i in range(self.n_steps):
            # Make a full step for the position
            _, mom_grad = self.get_grad(hamiltonian_func,pos,mom)
            pos.data +=  self.step_size * mom_grad
            # Make a full step for the momentum, except at end of trajectory
            if (i<self.n_steps-1):
                pos_grad, _ = self.get_grad(hamiltonian_func,pos,mom)
                mom.data -=  self.step_size * pos_grad
        pos_grad, _ = self.get_grad(hamiltonian_func,pos,mom)
        mom.data -= 0.5 * self.step_size * pos_grad

        return pos.data.numpy(), mom.data.numpy()