import numpy as np
import torch as th
from .utilities import np2th, th2np
import os
import json

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
        self.fit_time = None
        self.vacf = None
        self.Ivcf = None
        self.diff = None

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

    def init_statistics(self, nmax, isotropic=True):
        if self.traj is None:
            raise ValueError('Trajectory is not set yet.')
        if self.traj.naxis == 2:
            self.vacf = self.traj.compute_vacf(nmax)
            self.Ivcf = self.traj.compute_Ivcf(nmax)
            msd = self.traj.compute_msd(nmax)
            self.diff = self.traj.compute_diffusivity_half_grid()
        else:
            if isotropic is False:
                raise NotImplementedError('Anisotropic memory kernel for coarse-grained particle is not implemented yet.')
            vacf = self.traj.compute_vacf(nmax)
            Ivcf = self.traj.compute_Ivcf(nmax)
            msd = self.traj.compute_msd(nmax)
            diff = self.traj.compute_diffusivity_half_grid()
            self.vacf= vacf.mean(-1)
            self.Ivcf= Ivcf.mean(-1)
            self.diff= diff.mean(-1)
        return self.vacf, self.Ivcf, self.diff

    def train(self, fit_time, niter, lr_coef=0.0001, lr_tau=0.001, gamma=0.99, print_freq=100, save_freq=1000, save_dir='./gle_paras'):
        traj = self.traj
        dt = traj.dt
        self.fit_time = fit_time
        nmax = int(fit_time / dt)
        kbT = traj.kbT
        mass = traj.mass
        ndim = self.model.ndim

        ## get the statistics from trajectory data
        vacf, Ivcf, diff = self.init_statistics(nmax)
        ## make the diffusivity matrix
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
                if os.path.exists(save_dir) == False:
                    os.makedirs(save_dir)
                self.save_model(save_dir, idx)
                # self.model.save('./gle_paras/model_iter_{}.json'.format(idx))
        return
            
    def validate(self, val_time):
        traj = self.traj
        dt = traj.dt
        nmax = int(val_time / dt)
        kbT = traj.kbT
        mass = traj.mass
        ndim = self.model.ndim

        ## get the statistics from trajectory data
        vacf, Ivcf, diff = self.init_statistics(nmax)
        ## make the diffusivity matrix
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
    
    def save_model(self, save_dir, idx):
        model = self.model
        noise_coef = th2np(th.cat([model.noise_coef_cos, model.noise_coef_sin], -1))   # (ndim, 2*nmodes)
        mem_coef =   th2np(th.cat([model.mem_coef_cos,   model.mem_coef_sin],   -1))   # (ndim, 2*nmodes)
        gle_dict = {
            'kbT': model.kbT,
            'taus':  th2np(model.get_mem_taus()).tolist(),
            'freqs': th2np(model.get_mem_freqs()).tolist(),
            'noise_coef': noise_coef.tolist(),
            'mem_coef': mem_coef.tolist(),
            'mass': th2np(self.traj.mass).tolist(),
            'transform_matrix': th2np(self.traj.transform_matrix).tolist() if self.traj.transform_matrix is not None else None,
            'fit_time': self.fit_time,
        }
        save_path = os.path.join(save_dir, 'model_iter_{}.json'.format(idx))
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(gle_dict, f, ensure_ascii=False, indent=4)
        return