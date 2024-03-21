import numpy as np
import torch as th
from .utilities import np2th, th2np
from .decay_fourier import *
import json

 
class AIGLE(th.nn.Module):
    '''
    The class for the AIGLE model.
    '''
    def __init__(self, ndim, kbT=1, ntau=4, nfreq=4, mu=5, taus=None):
        super().__init__()
        '''
        Args:
            ndim: int, the number of dimensions of the collective variables
            ntau: int, the number of multi-scale relaxation times
            nfreq: int, the number of frequencies in the decay-Fourier series associated with each relaxation time
            taus: 1d array, initial guess of the multi-scale relaxation times
        '''
        ## get the fixed parameters
        self.ndim = ndim       # n
        self.ntau = ntau       # J
        self.nfreq = nfreq     # L
        self.mu = mu           # mu
        self.nmodes = ntau * nfreq  # JL
        self.kbT = kbT

        ## get the initial guess of the multi-scale relaxation time and store its logarithm
        if taus is None:
            _log_taus = np2th(np.arange(ntau) - 2.0) * 2.3 
        else:
            if taus.min() <= 0:
                raise ValueError('taus should be positive')
            if taus.ndim != 1:
                raise ValueError('taus should be 1d array')
            if ntau != taus.shape[0]:
                raise ValueError('ntau should be equal to the length of taus')
            if type(taus) is np.ndarray:
                _log_taus = np2th(np.log(taus))
            elif th.is_tensor(taus):
                _log_taus = th.log(np2th(th2np(taus)))
            else:
                raise TypeError('taus should be an array or a tensor')
        self.log_taus = th.nn.Parameter(_log_taus)

        ## the parameters of noise generator (\sigma_ijl), i=1,2,...,ndim, j=1,2,...,ntau, l=1,2,...,nfreq
        self.noise_coef_cos = th.nn.Parameter(th.zeros( ndim, self.nmodes)+0.0001)
        self.noise_coef_sin = th.nn.Parameter(th.zeros( ndim, self.nmodes)+0.0001)

        ## the parameters of memory kernel   (\phi_ijl),   i=1,2,...,ndim, j=1,2,...,ntau, l=1,2,...,nfreq
        ## these are not trainable, since they are calculated from the parameters of noise generator 
        self.register_buffer('mem_coef_cos', th.zeros( ndim, self.nmodes) )  # \phi_ijl
        self.register_buffer('mem_coef_sin', th.zeros( ndim, self.nmodes) )  # \phi_ijl
        
    def get_mem_taus(self):
        '''
        Returns:
            mem_taus: shape=(ntau*nfreq,)
        '''
        taus = th.exp(self.log_taus)
        mem_taus = th.tile(taus[:,None],(1,self.nfreq))
        mem_taus  = mem_taus.flatten()
        return mem_taus
        
    def get_mem_freqs(self):
        '''
        Returns:
            mem_freqs: shape=(ntau*nfreq,)
        '''
        taus = th.exp(self.log_taus)
        mem_freqs = 2 * np.pi / (self.mu * taus[:,None]) * np2th(np.arange(self.nfreq))[None,:]
        mem_freqs = mem_freqs.flatten()
        return mem_freqs
        
    # def get_mem_kernel(self, tgrid ):
    #     '''
    #     Returns:
    #         memory kernel: shape=(ndim, *)
    #     '''
    #     mem_taus = self.get_mem_taus()
    #     mem_freqs = self.get_mem_freqs()
    #     ## get the basis of xi-process and their cumulative sum
    #     mem_kernel_cos = get_decay_cos(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
    #     mem_kernel_sin = get_decay_sin(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
    #     mem_kernel  = (mem_kernel_cos[None,:,:] * self.mem_coef_cos[:,:,None]).sum(1)    ## (1,nmodes, nsteps)*(ndim, nmodes, 1)-> (ndim, nsteps) 
    #     mem_kernel += (mem_kernel_sin[None,:,:] * self.mem_coef_sin[:,:,None]).sum(1) 
    #     return mem_kernel  
    
    def get_M_tensor(self):
        """
        Returns:
            M_tensor: shape=(2,2, nmodes, nmodes)
        """
        mem_taus = self.get_mem_taus()
        mem_freqs = self.get_mem_freqs()
        inv_tau_mat = 1 / mem_taus[:,None] + 1 / mem_taus[None,:]
        Mcc = period_int_exp_cos_cos(inv_tau_mat, mem_freqs[:,None], mem_freqs[None,:])
        Msc = period_int_exp_sin_cos(inv_tau_mat, mem_freqs[:,None], mem_freqs[None,:])
        Mcs = period_int_exp_cos_sin(inv_tau_mat, mem_freqs[:,None], mem_freqs[None,:])
        Mss = period_int_exp_sin_sin(inv_tau_mat, mem_freqs[:,None], mem_freqs[None,:])
        M_tensor = th.stack([Mcc, Msc, Mcs, Mss], 0).reshape(2,2,self.nmodes, self.nmodes)  #  (2,2,JL,JL)
        return M_tensor

    def compute_mem_coef_from_2FDT(self, update_buffer=True):
        '''
        Compute the memory kernel coefficients from the parameters of noise generator with second fluctuation dissipation theorem.
        Returns:
            mem_coef_cos: shape=(ndim, nmodes)
            mem_coef_sin: shape=(ndim, nmodes)
        '''
        # mem_taus = self.get_mem_taus()
        # mem_freqs = self.get_mem_freqs()
        M_tensor = self.get_M_tensor()
        Mcc = M_tensor[0,0]
        Msc = M_tensor[0,1]
        Mcs = M_tensor[1,0]
        Mss = M_tensor[1,1]

        mem_coef_cos = []
        mem_coef_sin = []
        for i in range(self.ndim):
            sigma_cos_i = self.noise_coef_cos[i]
            sigma_sin_i = self.noise_coef_sin[i]
            
            phi_i_cos  = sigma_cos_i * (Mcc@sigma_cos_i)
            phi_i_cos += sigma_cos_i * (Mcs@sigma_sin_i)
            phi_i_cos += sigma_sin_i * (Msc@sigma_cos_i)
            phi_i_cos += sigma_sin_i * (Mss@sigma_sin_i)
            phi_i_cos = - phi_i_cos / self.kbT

            phi_i_sin  = -sigma_cos_i * (Msc@sigma_cos_i)
            phi_i_sin -= sigma_cos_i * (Mss@sigma_sin_i)
            phi_i_sin += sigma_sin_i * (Mcc@sigma_cos_i)
            phi_i_sin += sigma_sin_i * (Mcs@sigma_sin_i)
            phi_i_sin = - phi_i_sin / self.kbT
            mem_coef_cos.append(phi_i_cos)
            mem_coef_sin.append(phi_i_sin)
        mem_coef_cos = th.stack(mem_coef_cos, 0)
        mem_coef_sin = th.stack(mem_coef_sin, 0)
        if update_buffer:
            self.mem_coef_cos = mem_coef_cos.detach()
            self.mem_coef_sin = mem_coef_sin.detach()
        return mem_coef_cos, mem_coef_sin

    def compute_memory_kernel(self, tgrid):
        '''
        Compute the memory kernel on given time grid.
        '''
        mem_taus = self.get_mem_taus()
        mem_freqs = self.get_mem_freqs()
        mem_coef_cos, mem_coef_sin = self.compute_mem_coef_from_2FDT()
        basis_cos = get_decay_cos(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
        basis_sin = get_decay_sin(tgrid[None,:], mem_taus[:,None], mem_freqs[:,None])  ## (nmodes, nsteps)
        memory_kernel_cos = (mem_coef_cos[:,:,None] * basis_cos[None,:,:]).sum(1)  ## (ndim, nsteps)
        memory_kernel_sin = (mem_coef_sin[:,:,None] * basis_sin[None,:,:]).sum(1)  ## (ndim, nsteps)
        memory_kernel = memory_kernel_cos + memory_kernel_sin
        return memory_kernel

    
    def save(self, path):
        ## TODO: move this to trainer, save also mass and transform matrix.
        noise_coef = th2np(th.cat([self.noise_coef_cos, self.noise_coef_sin], -1))   # (ndim, 2*nmodes)
        mem_coef =   th2np(th.cat([self.mem_coef_cos,   self.mem_coef_sin],   -1))   # (ndim, 2*nmodes)
        gle_dict = {
            'kbT': self.kbT,
            'taus':  th2np(self.get_mem_taus()).tolist(),
            'freqs': th2np(self.get_mem_freqs()).tolist(),
            'noise_coef': noise_coef.tolist(),
            'mem_coef': mem_coef.tolist(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(gle_dict, f, ensure_ascii=False, indent=4)
    

    def load(self, path):
        raise NotImplementedError('This method is not implemented yet')
        # with open(path) as f:
        #     config = json.load(f)
        # mem_coef = np2th(np.array(config['mem_coef'])) ## (ndim, nmodes*2)
        # noise_coef = np2th(np.array(config['noise_coef'])) ## (ndim, nmodes*2)
        # nfreq = int(np.array(config['taus']).size / self.log_taus.shape[0])
        # taus = np2th(np.array(config['taus']))[::self.nfreq]
        # self.log_taus.data = th.log(taus)