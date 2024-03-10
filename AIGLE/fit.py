import numpy as np
import torch as th
from .utilities import np2th, th2np
from .decay_fourier import *
import json

 
class AIGLE(th.nn.Module):
    def __init__(self, dt, ndim, temp, v2avg, taus, nfreq=4):
        super().__init__()
        ## constants
        self.dt = dt
        self.ndim = ndim
        self.temp = temp # Kelvin
        self.v2avg = v2avg 
        if type(taus) is np.ndarray:
            _log_taus = np2th(np.log(taus))
        elif th.is_tensor(taus):
            _log_taus = th.log(taus)
        else:
            raise TypeError('taus should be an array or a tensor')
        self.ntaus = taus.shape[0]
        self.memory_in_tau = 5
        self.nfreq = nfreq
        self.nmodes = self.ntaus * nfreq
        ## parameters
        self.log_taus = th.nn.Parameter(_log_taus)
        self.noise_coef_cos = th.nn.Parameter(th.zeros( ndim, self.nmodes )+0.0001)
        self.noise_coef_sin = th.nn.Parameter(th.zeros( ndim, self.nmodes)+0.0001)
        self.register_buffer('mem_coef_cos', th.zeros( ndim, self.nmodes) )
        self.register_buffer('mem_coef_sin', th.zeros( ndim, self.nmodes) )
        ## optimizer
        self.optimizer = None
        self.scheduler = None
        
    @property
    def mem_taus(self):
        with th.no_grad():
            taus = th.exp(self.log_taus)
            nfreq = self.nfreq
            mem_taus = th.tile(taus[:,None],(1,nfreq))
            mem_taus  = mem_taus.flatten()
            return mem_taus
        
    @property
    def mem_freqs(self):
        with th.no_grad():
            taus = th.exp(self.log_taus)
            nfreq = self.nfreq
            mem_freqs = 2 * np.pi / (self.memory_in_tau * taus[:,None]) * np2th(np.arange(nfreq))[None,:]
            mem_freqs = mem_freqs.flatten()
            return mem_freqs
        
    def init_optimizer(self, lr_coef=0.0001, lr_tau=0.001, gamma=0.99):
        self.optimizer = th.optim.Adam([
                                {'params': [self.noise_coef_cos, self.noise_coef_sin,]},
                                {'params': [self.log_taus], 'lr': lr_tau},
                            ], lr=lr_coef)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=gamma)
        
    def get_mem_kernel(self, tgrid ):
        """
        Returns:
            memory kernel: shape=(ndim, nsteps)
        """
        with th.no_grad():
            mem_freqs = self.mem_freqs
            mem_taus = self.mem_taus
            ## get the basis of xi-process and their cumulative sum
            mem_kernel_cos = get_decay_cos(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
            mem_kernel_sin = get_decay_sin(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
            mem_kernel  = (mem_kernel_cos[None,:,:] * self.mem_coef_cos[:,:,None]).sum(1)    ## (1,nmodes, nsteps)*(ndim, nmodes, 1)-> (ndim, nsteps) 
            mem_kernel += (mem_kernel_sin[None,:,:] * self.mem_coef_sin[:,:,None]).sum(1) 
            return mem_kernel  
    
    def _compute_corr_rr(self, noise_coef_cos, noise_coef_sin, Mcc, Msc, Mcs, Mss):
        corr_rr_spectrum_cos  = noise_coef_cos * (Mcc@noise_coef_cos)
        corr_rr_spectrum_cos += noise_coef_cos * (Mcs@noise_coef_sin)
        corr_rr_spectrum_cos += noise_coef_sin * (Msc@noise_coef_cos)
        corr_rr_spectrum_cos += noise_coef_sin * (Mss@noise_coef_sin)

        corr_rr_spectrum_sin  = -noise_coef_cos * (Msc@noise_coef_cos)
        corr_rr_spectrum_sin -= noise_coef_cos * (Mss@noise_coef_sin)
        corr_rr_spectrum_sin += noise_coef_sin * (Mcc@noise_coef_cos)
        corr_rr_spectrum_sin += noise_coef_sin * (Mcs@noise_coef_sin)
        return corr_rr_spectrum_cos, corr_rr_spectrum_sin
    
    def _compute_kernel_from_2FDT(self, tgrid, tgrid_shifted):
        """
        Returns:
            mem_coef_cos: shape=(ndim, nmodes)
            mem_coef_sin: shape=(ndim, nmodes)
            mem_kernel: shape=(ndim, nsteps)
            mem_kernel_cumsum: shape=(ndim, nsteps)
        """
        nfreq = self.nfreq
        taus = th.exp(self.log_taus)
        mem_freqs = 2 * np.pi / (self.memory_in_tau * taus[:,None]) * np2th(np.arange(nfreq))[None,:]
        mem_taus = th.tile(taus[:,None],(1,nfreq))
        mem_taus  = mem_taus.flatten()
        mem_freqs = mem_freqs.flatten()
        ## get the overlapping matrix of xi-process
        tau_mat = 1 / mem_taus[:,None] + 1 / mem_taus[None,:]
        Mcc = period_int_exp_cos_cos(tau_mat, mem_freqs[:,None], mem_freqs[None,:])
        Msc = period_int_exp_sin_cos(tau_mat, mem_freqs[:,None], mem_freqs[None,:])
        Mcs = period_int_exp_cos_sin(tau_mat, mem_freqs[:,None], mem_freqs[None,:])
        Mss = period_int_exp_sin_sin(tau_mat, mem_freqs[:,None], mem_freqs[None,:])

        ## get the basis of xi-process and their cumulative sum
        mem_kernel_cos = get_decay_cos(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
        mem_kernel_sin = get_decay_sin(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
        mem_kernel_cumsum_cos = get_decay_cos_cumsum(tgrid_shifted[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
        mem_kernel_cumsum_sin = get_decay_sin_cumsum(tgrid_shifted[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)

        mem_kernel_list = []
        mem_kernel_cumsum_list = []
        mem_coef_cos_list = []
        mem_coef_sin_list = []
        ## calculate the coeffcient of xi-process in the memory kernel, from 2FDT
        for idx in range(self.ndim):
            noise_coef_cos = self.noise_coef_cos[idx]
            noise_coef_sin = self.noise_coef_sin[idx]
            corr_rr_spectrum_cos, corr_rr_spectrum_sin = self._compute_corr_rr(noise_coef_cos, noise_coef_sin, Mcc, Msc, Mcs, Mss)
            mem_coef_cos = - corr_rr_spectrum_cos / self.v2avg[idx]
            mem_coef_sin = - corr_rr_spectrum_sin / self.v2avg[idx]

            ## calculate the memory kernel
            mem_kernel  = (mem_kernel_cos * mem_coef_cos[:,None]).sum(0) 
            mem_kernel += (mem_kernel_sin * mem_coef_sin[:,None]).sum(0) 
            mem_kernel_cumsum  = (mem_kernel_cumsum_cos * mem_coef_cos[:,None]).sum(0) 
            mem_kernel_cumsum += (mem_kernel_cumsum_sin * mem_coef_sin[:,None]).sum(0) 
            mem_kernel_list.append(mem_kernel)
            mem_kernel_cumsum_list.append(mem_kernel_cumsum)
            mem_coef_cos_list.append(mem_coef_cos.detach())
            mem_coef_sin_list.append(mem_coef_sin.detach())
        return th.stack(mem_coef_cos_list), th.stack(mem_coef_sin_list), th.stack(mem_kernel_list), th.stack(mem_kernel_cumsum_list)
    
    def train(self, niter, ref_mem_kernel_tgrid, ref_mem_kernel, verbose=True):
        optimizer = self.optimizer
        ##
        ref_mem_kernel_cumsum = th.cumsum(ref_mem_kernel, dim=1) * self.dt
        ref_mem_kernel_cumsum_tgrid =  ref_mem_kernel_tgrid + 0.5 * self.dt
        ##
        loss_curve = []
        logtau_curve = []
        for idx in range(niter):
            mem_coef_cos, mem_coef_sin, mem_kernel, mem_kernel_cumsum = self._compute_kernel_from_2FDT(ref_mem_kernel_tgrid, ref_mem_kernel_cumsum_tgrid)
            loss = ((mem_kernel_cumsum - ref_mem_kernel_cumsum )**2).mean()
            loss += ((mem_kernel[:, :10]  - ref_mem_kernel[:, :10] )**2).mean()
            ## backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.scheduler.step()
            ## bookkeeping
            loss_curve.append(th2np(loss))
            logtau_curve.append(th2np(self.log_taus).copy())
            self.mem_coef_cos = mem_coef_cos.detach().clone()
            self.mem_coef_sin = mem_coef_sin.detach().clone()    
            ## logging
            if verbose:
                if idx % int(niter//100) ==0:
                    print('iter={}, loss={} tau={}'.format(idx, th2np(loss), th2np(th.exp(self.log_taus))))
        
        return np.array(loss_curve), np.array(logtau_curve)
    
    def save(self, path):
        noise_coef = th2np(th.cat([self.noise_coef_cos, self.noise_coef_sin], -1)) 
        mem_coef = th2np(th.cat([self.mem_coef_cos, self.mem_coef_sin], -1)) 
        gle_dict = {
            'temp': self.temp,
            'taus': th2np(self.mem_taus).tolist(),
            'freqs': th2np(self.mem_freqs).tolist(),
            'noise_coef': noise_coef.tolist(),
            'mem_coef': mem_coef.tolist(),
            'average_v2': self.v2avg.tolist(),
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(gle_dict, f, ensure_ascii=False, indent=4)
    

    def load(self, path):
        with open(path) as f:
            config = json.load(f)
        mem_coef = np2th(np.array(config['mem_coef'])) ## (ndim, nmodes*2)
        noise_coef = np2th(np.array(config['noise_coef'])) ## (ndim, nmodes*2)
        nfreq = int(np.array(config['taus']).size / self.log_taus.shape[0])
        taus = np2th(np.array(config['taus']))[::self.nfreq]
        self.log_taus.data = th.log(taus)