import numpy as np
import torch as th
# from .utilities import np2th, th2np
 
class GLESimulator:
    def __init__(self, config, timestep=1, ndim=1, mass=None, buffer_size=40):
        ## system parameters
        self.energy_engine = None
        self.force_engine = None
        self.position_constraint = None
        self.dt = timestep
        self.mass = mass
        self.kbT = config['kbT'] ## energy unit is kbT
        self._step = 0
        self.buffer_size = buffer_size
        
        ## GLE parameters
        self.taus =  np.array(config['taus']) 
        self.alphas = np.array(config['freqs']) 
        self.ndim = ndim
        nmodes = self.alphas.shape[0]
        self.nmodes = nmodes
        ## constanst for integration
        self.z0_cos = np.exp(- 0.5 * self.dt / self.taus) * np.cos(0.5 * self.dt * self.alphas) 
        self.z0_sin = np.exp(- 0.5 * self.dt / self.taus) * np.sin(0.5 * self.dt * self.alphas)
        self.z1 = np.exp(- self.dt / self.taus) * np.cos(self.dt * self.alphas)
        self.z2 = np.exp(- self.dt / self.taus) * np.sin(self.dt * self.alphas)
        self.z0_cos = self.z0_cos[None,:] ## (1, nmodes)
        self.z0_sin = self.z0_sin[None,:] ## (1, nmodes)
        self.z1 = self.z1[None,:] ## (1, nmodes)
        self.z2 = self.z2[None,:] ## (1, nmodes)

        ## memory and noise coefficients
        self.mem_coef = np.array(config['mem_coef']) ## (ndim, nmodes*2)
        self.noise_coef = np.array(config['noise_coef']) ## (ndim, nmodes*2)
        assert self.mem_coef.shape == (ndim, nmodes*2)
        assert self.noise_coef.shape == (ndim, nmodes*2)    
        self.mem_coef_cos = self.mem_coef[:,:nmodes] ## (ndim, nmodes)
        self.mem_coef_sin = self.mem_coef[:,nmodes:] ## (ndim, nmodes)
        self.noise_coef_cos = self.noise_coef[:,:nmodes] ## (ndim, nmodes)
        self.noise_coef_sin = self.noise_coef[:,nmodes:] ## (ndim, nmodes)
        
   
        
        ## system configuration
        self.x =  np.zeros(ndim) 
        self.x_history =  np.zeros((buffer_size, ndim)) 
        self.p =  np.zeros(ndim)

        ## GLE variables
        #### memory force
        # self.Fv_tot = th.zeros_like(self.x)  ## total memory force, shape = (ndim)
        self.Fv_cos =  np.zeros((ndim,nmodes)) ## cos component of memory force, shape = (ndim, nmodes)
        self.Fv_sin = np.zeros_like(self.Fv_cos)  ## sin component of memory force, shape = (ndim, nmodes)
        #### noise
        # self.noise_tot = th.zeros_like(self.x)  ## total noise , shape = (ndim)
        self.noise_cos = np.zeros_like(self.Fv_cos)  ## cos component of noise, shape = (ndim, nmodes)
        self.noise_sin = np.zeros_like(self.Fv_cos)  ## sin component of noise, shape = (ndim, nmodes)

    def get_langevin_integrator(self, timestep):
        a = 1/self.taus
        b = self.alphas
        
        ## compute the friction coefficient
        friction_cos = a / (a**2 + b**2)
        friction_sin = b / (a**2 + b**2)
        friction_tot = self.mem_coef_cos * friction_cos[None,:] + self.mem_coef_sin * friction_sin[None,:]
        friction_tot = -friction_tot.sum(-1)
        friction_tot = np.clip(friction_tot, a_min=1e-3, a_max=None)

        simulator = LESimulator( friction=friction_tot.copy(), 
            timestep=timestep, ndim=self.ndim, kbT=self.kbT, mass=self.mass.copy())
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
        self.x = x0.copy()
        self.x_history = self.x_history * 0 + self.x[None,:]

    def update_memory(self):
        """
        Update each memory component and sum them up
        """
        # Fv_cos (ndim, nmodes) 
        _Fv_cos = self.z0_cos * self.p[:,None] * self.dt + self.z1 * self.Fv_cos - self.z2 * self.Fv_sin
        _Fv_sin = self.z0_sin * self.p[:,None] * self.dt + self.z1 * self.Fv_sin + self.z2 * self.Fv_cos
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
        whiteNoise = np.random.randn(self.ndim) * (1/self.dt)**0.5
        _noise_cos = self.z0_cos * whiteNoise[:,None] * self.dt + self.z1 * self.noise_cos - self.z2 * self.noise_sin
        _noise_sin = self.z0_sin * whiteNoise[:,None] * self.dt + self.z1 * self.noise_sin + self.z2 * self.noise_cos
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
        dp_dt = force + Fv_tot + np.sqrt(mass) * noise_tot
        self.p += dt * dp_dt
        ## A
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
                    self.x_history = np.concatenate([self.x[None,:] * 1.0, self.x_history[:-1], ], 0)
                    self._step += 1
            else:
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
        self.mass = mass  ## kbT / (nm/ps)**2
        self._step = 0
        self.friction = friction  # (ndim)
        self.a = np.exp( -timestep * friction )
        self.b = ( 1 - np.exp( -2 * timestep * friction ))**0.5
        
        ## system configuration
        self.x =  np.zeros(ndim) 
        self.x_history =  np.zeros((40, ndim)) 
        self.v =  np.zeros(ndim)

    def set_force_engine(self, f_func):
        self.force_engine = f_func 

    def set_energy_engine(self, e_func):
        self.energy_engine = e_func

    def set_constraint(self, position_constraint):
        self.position_constraint = position_constraint

    def set_position(self, x0 ):
        self.x = x0.copy()
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
            gaussian = np.random.randn(self.ndim)
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
                    self.x_history = np.concatenate([self.x[None,:] * 1.0, self.x_history[:-1], ], 0)
                    self._step += 1
            else:
                self._step += 1