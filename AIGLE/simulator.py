import numpy as np
import torch as th
from .utilities import np2th, th2np
 

class GLESimulator:
    def __init__(self, config, timestep=1, ndim=1, mass=None):
        ## system parameters
        self.temp = config['temp']
        self.force_engine = None
        self.position_constraint = None
        self.dt = timestep
        self.mass = np2th(mass)
        self.kbT = 1 ## energy unit is kbT
        self._step = 0
        
        ## GLE parameters
        self.taus = np2th(np.array(config['taus']))
        self.ws = np2th(np.array(config['freqs']))
        self.ndim = ndim
        nmodes = self.ws.shape[0]
        self.nmodes = nmodes
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
        self.v = np2th(np.zeros(ndim))

        ## GLE variables
        #### memory force
        self.Fv_tot = th.zeros_like(self.x)  ## total memory force, shape = (ndim)
        self.Fv_cos = np2th(np.zeros((ndim,nmodes)))  ## cos component of memory force, shape = (ndim, nmodes)
        self.Fv_sin = th.zeros_like(self.Fv_cos)  ## sin component of memory force, shape = (ndim, nmodes)
        #### noise
        self.noise_tot = th.zeros_like(self.x)  ## total noise , shape = (ndim)
        self.noise_cos = th.zeros_like(self.Fv_cos)  ## cos component of noise, shape = (ndim, nmodes)
        self.noise_sin = th.zeros_like(self.Fv_cos)  ## sin component of noise, shape = (ndim, nmodes)

    def get_langevin_integrator(self, timestep):
        a = 1/self.taus
        b = self.ws
        friction_cos = a / (a**2 + b**2)
        friction_sin = b / (a**2 + b**2)
        friction_tot = self.mem_coef_cos * friction_cos[None,:] + self.mem_coef_sin * friction_sin[None,:]
        friction_tot = -friction_tot.sum(-1).clone().detach()
        friction_tot = th.clamp(friction_tot, min=1e-3)
        ### TODOL: fix cuda device issue
        return LESimulator( self.temp, friction=friction_tot, 
            timestep=timestep, ndim=self.ndim, mass=self.mass.clone().detach())
    
    def set_force_engine(self, f_func):
        self.force_engine = f_func 
    
    def set_constraint(self, position_constraint):
        self.position_constraint = position_constraint
    
    def set_position(self, x0 ):
        self.x = x0.to(device=self.x.device) if th.is_tensor(x0) else np2th(x0)
        
    def sumFv(self):
        """
        Update the total memory force Fv
        """
        self.Fv_tot = (self.Fv_cos * self.mem_coef_cos).sum(-1)
        self.Fv_tot += (self.Fv_sin * self.mem_coef_sin).sum(-1)
    
    def sumNoise(self):
        """
        Update the total noise
        """
        self.noise_tot = (self.noise_cos * self.noise_coef_cos).sum(-1)
        self.noise_tot += (self.noise_sin * self.noise_coef_sin).sum(-1)
            
    def updateMemory(self):
        """
        Update each memory component
        """
        dt = self.dt
        taus = self.taus[None, :]
        ws = self.ws[None, :]

        self.Fv_cos += dt * self.v[:,None]
        self.Fv_cos += dt * ( - self.Fv_cos / taus - self.Fv_sin * ws )
        self.Fv_sin += dt * ( - self.Fv_sin / taus + self.Fv_cos * ws )
    
    def updateNoise(self):
        """
        Update each noise component
        """
        dt = self.dt
        taus = self.taus[None, :]
        ws = self.ws[None, :]
        whiteNoise = th.randn_like(self.x) * (1/dt)**0.5
        
        self.noise_cos += dt * whiteNoise[:,None]
        self.noise_cos += dt * (- self.noise_cos / taus - self.noise_sin * ws )
        self.noise_sin += dt * (- self.noise_sin / taus + self.noise_cos * ws )
        
    # def applyConstrainVelocities(self):
    #     self.v -= (self.v * self.mass).sum() / self.mass.sum()
        
    def applyConstrainPositions(self):
        # self.x = th.clamp(self.x, min=np2th(np.array([0.16, 0.15])), max=np2th(np.array([1.63, 1.32])))
        # self.x = th.clamp(self.x, min=np2th(np.array([0.12, 0.10])), max=np2th(np.array([1.55, 1.3])))
        # self.x = th.clamp(self.x, min=0.25, max=1.25)
        self.x = self.position_constraint(self.x)
        
    def get_instant_temp(self):
        kinetic = 0.5 * (self.mass * self.v * self.v)
        instant_temp = 2 * kinetic / self.kbT * self.temp 
        return instant_temp
    
    # def step(self, n):
    #     """
    #     This one allows larger time step, why??
    #     scheme: ABOBA -> BOBAA
    #     B: x->x+vdt/2
    #     O: v->v+(F/m+Fv+noise)dt
    #     AA: update Fv and noise by dt
    #     """
    #     dt = self.dt
    #     for idx in range(n):
    #         aforce = self.force_engine(self.x) / self.mass
    #         # self.applyConstrainVelocities()
    #         ## AA
    #         self.updateMemory()
    #         self.updateNoise()
    #         ## B
    #         self.x += 0.5 * dt * self.v
    #         ## O
    #         self.sumNoise()
    #         self.sumFv()
    #         self.v += dt * aforce
    #         self.v += dt * self.Fv_tot
    #         self.v += dt * self.noise_tot
    #         ## B
    #         self.x += 0.5 * dt * self.v
    #         if self.position_constraint is not None:
    #             self.applyConstrainPositions()
    #         self._step += 1

    def step(self, n):
        """
        leap frog
        """
        dt = self.dt
        for idx in range(n):
            ## A
            self.x += 0.5 * dt * self.v
            ## 0
            aforce = self.force_engine(self.x) / self.mass
            self.updateMemory()
            self.updateNoise()
            self.sumNoise()
            self.sumFv()
            self.v += dt * aforce
            self.v += dt * self.Fv_tot
            self.v += dt * self.noise_tot
            ## A
            self.x += 0.5 * dt * self.v
            if self.position_constraint is not None:
                self.applyConstrainPositions()
            self._step += 1
                                
    def _step(self, n):
        """
        scheme: ABOBA -> AABOB
        B: x->x+vdt/2
        A: v->v+0.5(F/m)dt
        O: update Fv and noise by dt
        """
        dt = self.dt
        for idx in range(n):
            aforce = self.force_engine(self.x) / self.mass
            # AA
            self.v += dt * aforce
            ## B
            self.x += 0.5 * dt * self.v
            ## O
            self.updateMemory()
            self.updateNoise()
            self.sumNoise()
            self.sumFv()
            self.v += dt * self.Fv_tot
            self.v += dt * self.noise_tot
            ## B
            self.x += 0.5 * dt * self.v
            # self.applyConstrainPositions()
            if self.position_constraint is not None:
                self.applyConstrainPositions()
            self._step += 1

class LESimulator:
    def __init__(self, temp, friction, timestep=1, ndim=1, mass=None):
        ## system parameters
        self.dt = timestep
        self.temp = temp
        self.ndim = ndim
        self.force_engine = None
        self.position_constraint = None
        self.mass = np2th(mass)  ## kbT / (nm/ps)**2
        self.kbT = 1 ## energy unit is kbT
        self._step = 0
        self.friction = friction  # (ndim)
        self.a = th.exp( -timestep * friction )
        self.b = ( 1 - th.exp( -2 * timestep * friction ))**0.5
        
        ## system configuration
        self.x = np2th(np.zeros(ndim))
        self.v = np2th(np.zeros(ndim))

    def set_force_engine(self, f_func):
        self.force_engine = f_func 
        
    def set_constraint(self, position_constraint):
        self.position_constraint = position_constraint
        
    def set_position(self, x0 ):
        self.x = x0.to(device=self.x.device) if th.is_tensor(x0) else np2th(x0)
        
    # def applyConstrainVelocities(self):
    #     self.v -= (self.v * self.mass).sum() / self.mass.sum()
        
    def applyConstrainPositions(self):
        self.x = self.position_constraint(self.x)
        
    def get_instant_temp(self):
        kinetic = 0.5 * (self.mass * self.v * self.v).mean()
        instant_temp = 2 * kinetic / self.kbT * self.temp 
        return instant_temp
    
    def step(self, n):
        dt = self.dt
        for idx in range(n):
            aforce = self.force_engine(self.x) / self.mass
            gaussian = th.randn_like(self.x)
            self.v += dt * aforce
            self.x += 0.5 * dt * self.v
            self.v = self.a * self.v + self.b * gaussian * (self.kbT/self.mass)**0.5
            self.x += 0.5 * dt * self.v
            if self.position_constraint is not None:
                self.applyConstrainPositions()
            self._step += 1