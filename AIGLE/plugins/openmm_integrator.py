import numpy as np
import openmm
from openmm import unit
# from utilities import *



class LEIntegrator(openmm.CustomIntegrator):
    """
    The integrator for langevin equations (LE): ma = F - friction*v + white_noise
    For each particle, assume the memory and noise effects are isotropic.
    """
    def __init__(self, temperature, timestep,  friction):
        super(LEIntegrator, self).__init__(timestep)
        """
        Args:
            temperature (float or openmm.unit.Quantity): in unit of Kelvin
            timestep (float  or openmm.unit.Quantity): in unit of picosecond
            friction (numpy.array): shape=(nparticle), (value only) in unit of 1/picosecond
        """
        ## check inputs
        if type(timestep)  is unit.Quantity:
            dt = timestep.value_in_unit(unit.picosecond)
        else:
            dt = timestep
        kb = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA 
        if type(temperature) is unit.Quantity:
            kbT = (kb*temperature).value_in_unit(unit.kilojoule_per_mole)  #kJ/mol
        else:
            kbT = (kb*temperature*unit.kelvin).value_in_unit(unit.kilojoule_per_mole)  #kJ/mol
        if type(friction) is unit.Quantity:
            self.friction = friction.value_in_unit(unit.picosecond**-1)
        else:
            self.friction = friction
        self.friction = np.repeat(self.friction.reshape(-1,1),3,-1) ## (nparticle, 3)
        ## set variables
        self.addPerDofVariable("a", 0)
        self.setPerDofVariableByName("a", np.exp(-self.friction*dt))
        self.addPerDofVariable("b", 0)
        self.setPerDofVariableByName("b", np.sqrt(1-np.exp(-2*self.friction*dt)))
        self.addGlobalVariable("_kT", kbT)
        self.addPerDofVariable("x1", 0)
        
        ## set integration scheme
        self.addUpdateContextState()
        self.addConstrainVelocities()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + gaussian*b*sqrt(_kT/m)")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")

class GLEIntegrator(openmm.CustomIntegrator):
    """
    The integrator for generalized langevin equations (GLE): ma = F + \int K(t-s) v(s) ds + n(t)
    For each particle, assume the memory and noise effects are isotropic.
    """
    def __init__(self, config, timestep ):
        """
        Args:
            config: a dictionary containing the GLE parameters
            timestep (float): in unit of picosecond
        """
        super(GLEIntegrator, self).__init__(timestep)
        
        ################ Loading GLE Parameters ################
        self.taus =  np.array(config['taus']) 
        self.alphas =  np.array(config['freqs']) 
        self.nmodes = self.alphas.shape[0]
        ## constanst for integration
        self.z0_cos = np.exp(- 0.5 * timestep / self.taus) * np.cos(0.5 * timestep * self.alphas) 
        self.z0_sin = np.exp(- 0.5 * timestep / self.taus) * np.sin(0.5 * timestep * self.alphas)
        self.z1 = np.exp(- timestep / self.taus) * np.cos(timestep * self.alphas)
        self.z2 = np.exp(- timestep / self.taus) * np.sin(timestep * self.alphas)

        ## memory and noise coefficients
        self.mem_coef = np.array(config['mem_coef']) ## (ndim, nmodes*2)
        self.noise_coef = np.array(config['noise_coef']) ## (ndim, nmodes*2)
        self.nparticle = self.mem_coef.shape[0]
        assert self.mem_coef.shape == (self.nparticle, self.nmodes*2)
        assert self.noise_coef.shape == (self.nparticle, self.nmodes*2)       

        ################ Add Variables in OPENMM ################
        ## add CustomIntegrator global variables
        for i in range(self.nmodes):
            self.addGlobalVariable('Taus{:d}'.format(i), self.taus[i])
            self.addGlobalVariable('Freqs{:d}'.format(i), self.alphas[i])
            self.addGlobalVariable('Z0Cos{:d}'.format(i), self.z0_cos[i])
            self.addGlobalVariable('Z0Sin{:d}'.format(i), self.z0_sin[i])
            self.addGlobalVariable('Z1{:d}'.format(i), self.z1[i])
            self.addGlobalVariable('Z2{:d}'.format(i), self.z2[i])
        
        ## add CustomIntegrator per-Dof Variables
        ## assuming isotropic memery effects, replicate the memory&noise coefficients for x,y,z
        _mem_coef = np.tile(self.mem_coef, 3).reshape(self.nparticle, 3, -1)
        _noise_coef = np.tile(self.noise_coef, 3).reshape(self.nparticle, 3, -1)
        self.addPerDofVariable('Fv',0)  ## total memory force
        self.addMemoryComponents( _mem_coef[:,:,:self.nmodes], _mem_coef[:,:,self.nmodes:]) ## components of memory force
        self.addPerDofVariable('Noise',0) ## total noise
        self.addNoiseComponents( _noise_coef[:,:,:self.nmodes], _noise_coef[:,:,self.nmodes:]) ## components of noise
        
        ## buffer
        self.addPerDofVariable('BufferCos', 0)
        self.addPerDofVariable('BufferSin', 0)
        self.addPerDofVariable('whiteNoise',0)
        self.addPerDofVariable("x1", 0)
        
        ## set integration scheme
        self.setIntegrationScheme()

    def get_langevin_integrator(self, temperature,  timestep, epsilon=1e-3):
        """
        Args:
            temperature (float): in unit of Kelvin
            timestep (float): in unit of picosecond
            epsilon (float): the minimum value of the friction coefficient, in unit of 1/picosecond. 
                            epsilon should be positive to avoid negative friction coefficient derived from GLE.
        Returns:
            a Langevin Equation integrator with the instantaneous friction derived from the GLE integrator
        """
        ## get parameters
        a = 1/self.taus
        b = self.alphas
        mem_coef_cos = self.mem_coef[:,:self.nmodes]
        mem_coef_sin = self.mem_coef[:,self.nmodes:]
        
        ## compute the friction coefficient
        friction_cos = a / (a**2 + b**2)
        friction_sin = b / (a**2 + b**2)
        friction_tot = mem_coef_cos * friction_cos[None,:] + mem_coef_sin * friction_sin[None,:]
        friction_tot = -friction_tot.sum(-1)
        friction_tot = np.clip(friction_tot, a_min=epsilon,a_max=None)
        return LEIntegrator( temperature, timestep, friction=friction_tot )
    
    def addMemoryComponents(self, mem_cf_cos, mem_cf_sin):
        """
        Adds memory components to the integrator.
        FvCos[i] and Fvsin[i] value only, in the unit of length (nm).
        MemCoefCos[i] and MemCoefSin[i] value only, in the the unit of (1/ps)^2.
        Args:
            mem_cf_cos (np.array): shape=(nparticle,3, nmodes), (value only) in unit of nm (1/ps)^2
            mem_cf_sin (np.array): shape=(nparticle,3, nmodes), (value only) in unit of (1/ps)^2
        """
        for i in range(self.nmodes):
            ## the memory components
            self.addPerDofVariable('FvCos{:d}'.format(i), 0)
            self.addPerDofVariable('FvSin{:d}'.format(i), 0)
            ## the weight of the memory components in Fv
            self.addPerDofVariable('MemCoefCos{:d}'.format(i), 0 )
            self.setPerDofVariableByName('MemCoefCos{:d}'.format(i), mem_cf_cos[...,i] )
            self.addPerDofVariable('MemCoefSin{:d}'.format(i), 0 )
            self.setPerDofVariableByName('MemCoefSin{:d}'.format(i), mem_cf_sin[...,i])
    
    def addNoiseComponents(self,  noise_cf_cos, noise_cf_sin):
        """
        Adds noise components to the integrator.
        NoiseCos[i] and NoiseSin[i] have the unit of length (nm).
        NoiseCoefCos[i] and NoiseCoefSin[i] have the unit of (1/ps)^2.
        Args:
            noise_cf_cos (np.array): shape=(nparticle,3, nmodes), (value only) in unit of (1/ps)^2
            noise_cf_sin (np.array): shape=(nparticle,3, nmodes), (value only) in unit of (1/ps)^2
        """
        for i in range(self.nmodes):
            ## the xi process
            self.addPerDofVariable('NoiseCos{:d}'.format(i),0)
            self.addPerDofVariable('NoiseSin{:d}'.format(i),0)
            ## the weight of the xi process in noise
            self.addPerDofVariable('NoiseCoefCos{:d}'.format(i), 0 )
            self.setPerDofVariableByName('NoiseCoefCos{:d}'.format(i), noise_cf_cos[...,i] )
            self.addPerDofVariable('NoiseCoefSin{:d}'.format(i), 0 )
            self.setPerDofVariableByName('NoiseCoefSin{:d}'.format(i), noise_cf_sin[...,i])

    def updateMemory(self):
        """
        Update each memory component. Then sum up the memory force.
        """
        for i in range(self.nmodes):
            self.addComputePerDof("BufferCos", "Z1{:d}*FvCos{:d} - Z2{:d}*FvSin{:d}".format(i,i,i,i))
            self.addComputePerDof("BufferSin", "Z1{:d}*FvSin{:d} + Z2{:d}*FvCos{:d}".format(i,i,i,i))
            self.addComputePerDof("FvCos{:d}".format(i), 
                                  "BufferCos + dt*Z0Cos{:d}*v*m".format(i))
            self.addComputePerDof("FvSin{:d}".format(i),
                                  "BufferSin + dt*Z0Sin{:d}*v*m".format(i))
        ## sum up the force
        self.addComputePerDof("Fv", "Fv*0")
        for i in range(self.nmodes):
            self.addComputePerDof("Fv", 
                                  "Fv + MemCoefCos{:d}*FvCos{:d} + MemCoefSin{:d}*FvSin{:d}".format(i,i,i,i))
        
    def updateNoise(self):
        """
        Update each noise component. Then sum up the noise.
        """
        self.addComputePerDof("whiteNoise", "sqrt(1/dt) * gaussian")
        for i in range(self.nmodes):
            ## cosine xi-process will take in the white noise scaled by sqrt(kBT/tau/m/dt)
            self.addComputePerDof("BufferCos", "Z1{:d}*NoiseCos{:d} - Z2{:d}*NoiseSin{:d}".format(i,i,i,i))
            self.addComputePerDof("BufferSin", "Z1{:d}*NoiseSin{:d} + Z2{:d}*NoiseCos{:d}".format(i,i,i,i))
            self.addComputePerDof("NoiseCos{:d}".format(i),
                                  "BufferCos + dt*Z0Cos{:d}*whiteNoise".format(i))
            self.addComputePerDof("NoiseSin{:d}".format(i),
                                  "BufferSin + dt*Z0Sin{:d}*whiteNoise".format(i))
        ## sum up the noise
        self.addComputePerDof("Noise", "Noise*0")
        for i in range(self.nmodes):
            self.addComputePerDof("Noise", 
                                  "Noise + NoiseCoefCos{:d}*NoiseCos{:d} + NoiseCoefSin{:d}*NoiseSin{:d}".format(i,i,i,i))
                        
    def setIntegrationScheme(self):
        """
        leap frog
        """
        self.addConstrainVelocities()
        ## update position
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addUpdateContextState()
        ## update memory force and noise
        self.updateMemory()
        self.updateNoise()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addComputePerDof("v", "v + dt*Fv/m")
        self.addComputePerDof("v", "v + dt*Noise/sqrt(m)")
        ## update position
        self.addComputePerDof("x", "x + 0.5*dt*v")
        ## constraint positions
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")

 