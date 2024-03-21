import numpy as np
import torch as th
from .utilities import np2th, th2np
import torch.jit as jit

@jit.script
def update_memory_jit(Fv_cos, Fv_sin, p, taus, alphas, dt, mem_coef_cos, mem_coef_sin):
    _Fv_cos = Fv_cos + dt * p[:,None]
    _Fv_cos += dt * ( - Fv_cos / taus - Fv_sin * alphas )
    _Fv_sin = Fv_sin + dt * ( - Fv_sin / taus + Fv_cos * alphas )
    
    ## get the total memory force Fv
    Fv_tot = (_Fv_cos * mem_coef_cos).sum(-1)
    Fv_tot += (_Fv_sin * mem_coef_sin).sum(-1)
    return _Fv_cos, _Fv_sin, Fv_tot

@jit.script
def update_noise_jit(noise_cos, noise_sin, whiteNoise, taus, alphas, dt, noise_coef_cos, noise_coef_sin):
    """
    Update each noise component and sum them up
    """    
    _noise_cos = noise_cos + dt * whiteNoise[:,None]
    _noise_cos += dt * (- noise_cos / taus - noise_sin * alphas )
    _noise_sin = noise_sin + dt * (- noise_sin / taus + noise_cos * alphas )
    
    ## get the total noise
    noise_tot = (noise_cos * noise_coef_cos).sum(-1)
    noise_tot += (noise_sin * noise_coef_sin).sum(-1)
    return _noise_cos, _noise_sin, noise_tot


class GLESimulator:
    def __init__(self, config, timestep=1, ndim=1, mass=None, buffer_size=40):
        ## system parameters
        self.dev = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.energy_engine = None
        self.force_engine = None
        self.position_constraint = None
        self.dt = th.tensor(timestep, dtype=th.float32, device=self.dev)
        # self.dt = timestep
        self.mass = np2th(mass)
        self.kbT = config['kbT'] ## energy unit is kbT
        self._step = 0
        self.buffer_size = buffer_size
        
        ## GLE parameters
        self.taus = np2th(np.array(config['taus']))
        self.alphas = np2th(np.array(config['freqs']))
        self.ndim = ndim
        nmodes = self.alphas.shape[0]
        self.nmodes = nmodes
        ## constanst for integration
        self.z_mat = self.get_z_matrix()
        self.z0 = (1 - self.dt / self.taus[None, :])
        self.z1 = self.dt * self.alphas[None, :]
        ## memory and noise coefficients
        self.mem_coef = np2th(np.array(config['mem_coef'])) ## (ndim, nmodes*2)
        self.mem_coef_cos = self.mem_coef[:,:nmodes] ## (ndim, nmodes)
        self.mem_coef_sin = self.mem_coef[:,nmodes:] ## (ndim, nmodes)
        self.noise_coef = np2th(np.array(config['noise_coef'])) ## (ndim, nmodes*2)
        self.noise_coef_cos = self.noise_coef[:,:nmodes] ## (ndim, nmodes)
        self.noise_coef_sin = self.noise_coef[:,nmodes:] ## (ndim, nmodes)
        
        
        assert self.mem_coef.shape == (ndim, nmodes*2)
        assert self.noise_coef.shape == (ndim, nmodes*2)       
        
        ## system configuration
        self.x = np2th(np.zeros(ndim))
        self.x_history = np2th(np.zeros((buffer_size, ndim)))
        self.p = np2th(np.zeros(ndim))

        ## GLE variables
        #### memory force
        # self.Fv_tot = th.zeros_like(self.x)  ## total memory force, shape = (ndim)
        self.Fv_cos = np2th(np.zeros((ndim,nmodes)))  ## cos component of memory force, shape = (ndim, nmodes)
        self.Fv_sin = th.zeros_like(self.Fv_cos)  ## sin component of memory force, shape = (ndim, nmodes)
        #### noise
        # self.noise_tot = th.zeros_like(self.x)  ## total noise , shape = (ndim)
        self.noise_cos = th.zeros_like(self.Fv_cos)  ## cos component of noise, shape = (ndim, nmodes)
        self.noise_sin = th.zeros_like(self.Fv_cos)  ## sin component of noise, shape = (ndim, nmodes)

    def get_z_matrix(self):
        dt = self.dt
        z_matrix = []
        for idx in range(self.nmodes):
            tau = self.taus[idx]
            alpha = self.alphas[idx]
            m = th.tensor([[-dt / tau, -dt * alpha],
                           [dt * alpha, -dt / tau]], 
                           device=tau.device, dtype=tau.dtype)
            expm = th.matrix_exp(m)
            z_matrix.append(expm)
        z_matrix = th.stack(z_matrix, 0)  ## (nmodes, 2, 2)
        return z_matrix

    def get_langevin_integrator(self, timestep):
        a = 1/self.taus
        b = self.alphas
        friction_cos = a / (a**2 + b**2)
        friction_sin = b / (a**2 + b**2)
        friction_tot = self.mem_coef_cos * friction_cos[None,:] + self.mem_coef_sin * friction_sin[None,:]
        friction_tot = -friction_tot.sum(-1).clone().detach()
        friction_tot = th.clamp(friction_tot, min=1e-3)

        simulator = LESimulator( friction=friction_tot.clone(), 
            timestep=timestep, ndim=self.ndim, kbT=self.kbT, mass=self.mass.clone())
        simulator.set_force_engine(self.force_engine)
        simulator.set_energy_engine(self.energy_engine)
        simulator.set_constraint(self.position_constraint)
        ### TODOL: fix cuda device issue
        return simulator
    
    def set_force_engine(self, f_func):
        self.force_engine = f_func 

    def set_energy_engine(self, e_func):
        self.energy_engine = e_func 
   
    def set_constraint(self, position_constraint):
        self.position_constraint = position_constraint

    def set_position(self, x0 ):
        self.x = x0.to(device=self.x.device) if th.is_tensor(x0) else np2th(x0)
        self.x_history = self.x_history * 0 + self.x[None,:]

    def update_memory(self):
        """
        Update each memory component and sum them up
        """
        # Fv_cos (ndim, nmodes),  z_mat : (nmodes, 2,2 )
        dt = self.dt
        z_mat = self.z_mat
        self.Fv_cos += self.p[:,None] * dt
        _Fv_cos = self.Fv_cos * z_mat[None,:,0,0] + self.Fv_sin * z_mat[None,:,0,1]
        _Fv_sin = self.Fv_cos * z_mat[None,:,1,0] + self.Fv_sin * z_mat[None,:,1,1]
        self.Fv_cos = _Fv_cos
        self.Fv_sin = _Fv_sin
        ## get the total memory force
        Fv_tot = (_Fv_cos * self.mem_coef_cos).sum(-1)
        Fv_tot += (_Fv_sin * self.mem_coef_sin).sum(-1)
        return Fv_tot
    
    def update_noise(self):
        """
        Update each noise component and sum them up
        """
        dt = self.dt
        z_mat = self.z_mat
        whiteNoise = th.randn_like(self.x) * (1/dt)**0.5
        self.noise_cos +=  whiteNoise[:,None] * dt
        _noise_cos = self.noise_cos * z_mat[None,:,0,0] + self.noise_sin * z_mat[None,:,0,1]
        _noise_sin = self.noise_cos * z_mat[None,:,1,0] + self.noise_sin * z_mat[None,:,1,1]
        self.noise_cos = _noise_cos
        self.noise_sin = _noise_sin
        ## get the total noise
        noise_tot = (_noise_cos * self.noise_coef_cos).sum(-1)
        noise_tot += (_noise_sin * self.noise_coef_sin).sum(-1)
        return noise_tot
        
    def applyConstrainPositions(self):
        self.x = self.position_constraint(self.x)
        
    def get_instant_temp(self):
        '''
        Returns the instant temperature in unit of kbT
        '''
        kinetic = 0.5 * (self.p * self.p / self.mass).mean()
        instant_temp = 2 * kinetic / self.kbT
        return instant_temp
    
    def update_state(self):
        """
        Update the state of the system
        """
        dt = self.dt
        mass = self.mass
        self.x += 0.5 * dt * self.p / mass
        force = self.force_engine(self.x) 
        Fv_tot = self.update_memory()
        noise_tot = self.update_noise()
        dp_dt = force + Fv_tot + th.sqrt(mass) * noise_tot
        self.p += dt * dp_dt
        ## A
        self.x += 0.5 * dt * self.p / mass
        return 
 
    def update_state_jit(self):
        """
        Update the state of the system
        """
        dt = self.dt
        mass = self.mass
        self.x += 0.5 * dt * self.p / mass
        force = self.force_engine(self.x) 
        
        Fv_cos, Fv_sin, Fv_tot = update_memory_jit(self.Fv_cos, 
                                                self.Fv_sin, 
                                                self.p, 
                                                self.taus[None, :], 
                                                self.alphas[None, :], 
                                                self.dt, 
                                                self.mem_coef_cos, 
                                                self.mem_coef_sin)
        self.Fv_cos = Fv_cos
        self.Fv_sin = Fv_sin

        whiteNoise = th.randn_like(self.x) * (1/dt)**0.5
        noise_cos, noise_sin, noise_tot = update_noise_jit(self.noise_cos,
                                                        self.noise_sin, 
                                                        whiteNoise, 
                                                        self.taus[None, :], 
                                                        self.alphas[None, :], 
                                                        self.dt, 
                                                        self.noise_coef_cos, 
                                                        self.noise_coef_sin)
        self.noise_cos = noise_cos
        self.noise_sin = noise_sin
        dp_dt = force + Fv_tot + th.sqrt(mass) * noise_tot
        self.p += dt * dp_dt
        self.x += 0.5 * dt * self.p / mass
        return 
        
    def step(self, n, energy_upper_bound=None):
        """
        leap frog
        """
        for idx in range(n):
            self.update_state()
            if self.position_constraint is not None:
                self.applyConstrainPositions()
            ## check energy constraint
            if energy_upper_bound is not None:
                energy = self.energy_engine(self.x)
                if energy > energy_upper_bound:
                    self.x = self.x_history[-1]
                    print('warning: step={}, enter unexplored region, reset to previous position'.format(self._step))
                    if self._step > self.x_history.shape[0]:
                        self._step -= (self.x_history.shape[0]-1)
                    else:
                        raise ValueError('Invalid initial position, please reset')
                else:
                    self.x_history = th.cat([self.x[None,:] * 1.0, self.x_history[:-1], ], dim=0)
                    self._step += 1
 
class LESimulator:
    def __init__(self, friction, timestep=1, ndim=1, kbT=1,  mass=None):
        ## system parameters
        self.dt = timestep
        self.kbT = kbT
        self.ndim = ndim
        self.force_engine = None
        self.energy_engine = None
        self.position_constraint = None
        self.mass = np2th(mass)  ## kbT / (nm/ps)**2
        self._step = 0
        self.friction = friction  # (ndim)
        self.a = th.exp( -timestep * friction )
        self.b = ( 1 - th.exp( -2 * timestep * friction ))**0.5
        
        ## system configuration
        self.x = np2th(np.zeros(ndim))
        self.x_history = np2th(np.zeros((40, ndim)))
        self.v = np2th(np.zeros(ndim))

    def set_force_engine(self, f_func):
        self.force_engine = f_func 

    def set_energy_engine(self, e_func):
        self.energy_engine = e_func

    def set_constraint(self, position_constraint):
        self.position_constraint = position_constraint

    def set_position(self, x0 ):
        self.x = x0.to(device=self.x.device) if th.is_tensor(x0) else np2th(x0)
        self.x_history = self.x_history *0 + self.x[None,:]
        
    def applyConstrainPositions(self):
        self.x = self.position_constraint(self.x)
        
    def get_instant_temp(self):
        '''
        Returns the instant temperature in unit of kbT
        '''
        kinetic = 0.5 * (self.v * self.v * self.mass).mean()
        instant_temp = 2 * kinetic / self.kbT 
        return instant_temp
    
    def step(self, n, energy_upper_bound=None):
        dt = self.dt
        for idx in range(n):
            force = self.force_engine(self.x)
            gaussian = th.randn_like(self.x)
            self.v += dt * force / self.mass
            self.x += 0.5 * dt * self.v
            self.v = self.a * self.v + self.b * gaussian * (self.kbT/self.mass)**0.5
            self.x += 0.5 * dt * self.v
            if self.position_constraint is not None:
                self.applyConstrainPositions()
            ## check energy constraint
            if energy_upper_bound is not None:
                energy = self.energy_engine(self.x)
                if energy > energy_upper_bound:
                    self.x = self.x_history[-1]
                    print('warning: step={}, enter unexplored region, reset to previous position'.format(self._step))
                    if self._step > self.x_history.shape[0]:
                        self._step -= (self.x_history.shape[0]-1)
                    else:
                        raise ValueError('Invalid initial position, please reset')
                else:
                    self.x_history = th.cat([self.x[None,:] * 1.0, self.x_history[:-1], ], dim=0)
                    self._step += 1