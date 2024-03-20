import numpy as np
import torch as th
from .utilities import np2th, th2np
import os

class Trainer:
    '''
    The class for fitting the AIGLE to given trajectories of collective variables.
    '''
    def __init__(self, model):
        ## optimizer
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.traj = None
    
    def init_optimizer(self, lr_coef=0.0001, lr_tau=0.001, gamma=0.99):
        self.optimizer = th.optim.Adam([
                                {'params': [self.model.noise_coef_cos, self.model.noise_coef_sin,]},
                                {'params': [self.model.log_taus], 'lr': lr_tau},
                            ], lr=lr_coef)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=gamma)
        return

    def set_traj(self, traj):
        self.traj = traj
        return

    def train(self, fit_time, niter, lr_coef=0.0001, lr_tau=0.001, gamma=0.99, print_freq=100, save_freq=1000):
        traj = self.traj
        dt = traj.dt
        nmax = int(fit_time / dt)
        kbT = traj.kbT
        mass = traj.mass
        ndim = self.model.ndim
        ## initialize data
        vacf = traj.compute_vacf(nmax)               # (nmax, ndim)    
        Ivcf = traj.compute_Ivcf(nmax)               # (nmax, ndim)
        msd  = traj.compute_msd(nmax)                # (nmax, ndim)
        diff = traj.compute_diffusivity_half_grid()  # (nmax-1, ndim)
        mat_diff = np2th(np.zeros((nmax-1, nmax-1, ndim)))
        for ii in range(nmax-1):
            mat_diff[ii, :ii+1] = th.flip(diff[ :ii+1], [0])
        impulse_mem = - kbT + mass * vacf - Ivcf  # (nmax, ndim)
        ## initialize optimizer and time grid on which loss is computed
        self.init_optimizer(lr_coef=lr_coef, lr_tau=lr_tau, gamma=gamma)
        lossfn = th.nn.MSELoss(reduction='mean')
        tgrid = (np2th(np.arange(nmax-1)) + 0.5) * dt
        ## Train loop
        for idx in range(niter):
            mem_kernel = self.model.compute_memory_kernel(tgrid)  ## (ndim, nmax-1)
            pred_impulse_mem = mat_diff * th.transpose(mem_kernel, 0, 1).unsqueeze(0)  ## (nmax-1, nmax-1, ndim)
            pred_impulse_mem = pred_impulse_mem.sum(1) * dt * mass[None,:]  ## (nmax-1, ndim)
            loss = lossfn(impulse_mem[1:], pred_impulse_mem) 
            ## backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            ## logging
            if idx % print_freq ==0:
                print('iter={}, loss={:.6f}, tau={}'.format(idx, th2np(loss), th2np(th.exp(self.model.log_taus.data))))
            if idx % save_freq ==0:
                if os.path.exists('./gle_paras') == False:
                    os.makedirs('./gle_paras')
                self.model.save('./gle_paras/model_iter_{}.json'.format(idx))
        return
            
    def validate(self, val_time):
        traj = self.traj
        dt = traj.dt
        nmax = int(val_time / dt)
        kbT = traj.kbT
        mass = traj.mass
        ndim = self.model.ndim
        ## initialize data
        vacf = traj.compute_vacf(nmax)               # (nmax, ndim)    
        Ivcf = traj.compute_Ivcf(nmax)               # (nmax, ndim)
        msd  = traj.compute_msd(nmax)                # (nmax, ndim)
        diff = traj.compute_diffusivity_half_grid()  # (nmax-1, ndim)
        mat_diff = np2th(np.zeros((nmax-1, nmax-1, ndim)))
        for ii in range(nmax-1):
            mat_diff[ii, :ii+1] = th.flip(diff[ :ii+1], [0])
        impulse_mem = - kbT + mass * vacf - Ivcf  # (nmax, ndim)
        with th.no_grad():
            ## get predicted impulse
            tgrid = (np2th(np.arange(nmax-1)) + 0.5) * dt
            mem_kernel = self.model.compute_memory_kernel(tgrid)  ## (ndim, nmax-1)
            pred_impulse_mem = mat_diff * th.transpose(mem_kernel, 0, 1).unsqueeze(0)  ## (nmax-1, nmax-1, ndim)
            pred_impulse_mem = pred_impulse_mem.sum(1) * dt * mass[None,:]  ## (nmax-1, ndim)
        return impulse_mem, pred_impulse_mem