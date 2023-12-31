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
        
        ## loading parameters
        self.mem_taus =  np.array(config['taus']) 
        self.mem_freqs =  np.array(config['freqs']) 
        self.nmodes = self.mem_freqs.shape[0]
        self.mem_coef = np.array(config['mem_coef'])  ## (nparticle, nmodes*2)
        self.noise_coef = np.array(config['noise_coef']) ## (nparticle, nmodes*2)
        self.nparticle = self.mem_coef.shape[0]
        assert self.mem_coef.shape == self.noise_coef.shape, "Dimension of memory and noise coefficient does not match"
        assert self.mem_coef.shape[1] == 2*self.nmodes, "The number of Memory coefficient for each particle does not match number of modes"
        
        ## add CustomIntegrator global variables
        for i in range(self.nmodes):
            self.addGlobalVariable('Taus{:d}'.format(i), self.mem_taus[i])
            self.addGlobalVariable('Freqs{:d}'.format(i), self.mem_freqs[i])
        
        ## add CustomIntegrator per-Dof Variables
        ## assuming isotropic memery effects, replicate the memory&noise coefficients for x,y,z
        _mem_coef = np.tile(self.mem_coef, 3).reshape(self.nparticle, 3, -1)
        _noise_coef = np.tile(self.noise_coef, 3).reshape(self.nparticle, 3, -1)
        self.addPerDofVariable('Fv',0)  ## total memory force
        self.addMemoryComponents( _mem_coef[:,:,:self.nmodes], _mem_coef[:,:,self.nmodes:]) ## components of memory force
        self.addPerDofVariable('Noise',0) ## total noise
        self.addNoiseComponents( _noise_coef[:,:,:self.nmodes], _noise_coef[:,:,self.nmodes:]) ## components of noise
        
        ## buffer
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
        a = 1/self.mem_taus
        b = self.mem_freqs
        mem_coef_cos = self.mem_coef[:,:self.nmodes]
        mem_coef_sin = self.mem_coef[:,self.nmodes:]
        noise_coef_cos = self.noise_coef[:,:self.nmodes]
        noise_coef_sin = self.noise_coef[:,self.nmodes:]
        
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
        CoefFvcos[i] and CoefFvsin[i] value only, in the the unit of (1/ps)^2.
        Args:
            mem_cf_cos (np.array): shape=(nparticle,3, nmodes), (value only) in unit of nm (1/ps)^2
            mem_cf_sin (np.array): shape=(nparticle,3, nmodes), (value only) in unit of (1/ps)^2
        """
        for i in range(self.nmodes):
            ## the memory components
            self.addPerDofVariable('FvCos{:d}'.format(i), 0)
            self.addPerDofVariable('FvSin{:d}'.format(i), 0)
            ## the weight of the memory components in Fv
            self.addPerDofVariable('CoefFvCos{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefFvCos{:d}'.format(i), mem_cf_cos[...,i] )
            self.addPerDofVariable('CoefFvSin{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefFvSin{:d}'.format(i), mem_cf_sin[...,i])
    
    def addNoiseComponents(self,  noise_cf_cos, noise_cf_sin):
        """
        Adds noise components to the integrator.
        NoiseCos[i] and NoiseSin[i] have the unit of length (nm).
        CoefNoiseCos[i] and CoefNoiseSin[i] have the unit of (1/ps)^2.
        Args:
            noise_cf_cos (np.array): shape=(nparticle,3, nmodes), (value only) in unit of (1/ps)^2
            noise_cf_sin (np.array): shape=(nparticle,3, nmodes), (value only) in unit of (1/ps)^2
        """
        for i in range(self.nmodes):
            ## the xi process
            self.addPerDofVariable('NoiseCos{:d}'.format(i),0)
            self.addPerDofVariable('NoiseSin{:d}'.format(i),0)
            ## the weight of the xi process in noise
            self.addPerDofVariable('CoefNoiseCos{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefNoiseCos{:d}'.format(i), noise_cf_cos[...,i] )
            self.addPerDofVariable('CoefNoiseSin{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefNoiseSin{:d}'.format(i), noise_cf_sin[...,i])

    def updateMemory(self):
        """
        Update each memory component. Then sum up the memory force.
        """
        for i in range(self.nmodes):
            self.addComputePerDof("FvCos{:d}".format(i), 
                                  "FvCos{:d} + dt*v".format(i))
            self.addComputePerDof("FvCos{:d}".format(i), 
                                  "FvCos{:d} + dt*( - FvCos{:d}/Taus{:d} - FvSin{:d}*Freqs{:d})".format(i,i,i,i,i))
            self.addComputePerDof("FvSin{:d}".format(i),
                                  "FvSin{:d} + dt*( - FvSin{:d}/Taus{:d} + FvCos{:d}*Freqs{:d})".format(i,i,i,i,i))
        ## sum up the force
        self.addComputePerDof("Fv", "Fv*0")
        for i in range(self.nmodes):
            self.addComputePerDof("Fv", 
                                  "Fv + CoefFvCos{:d}*FvCos{:d} + CoefFvSin{:d}*FvSin{:d}".format(i,i,i,i))
        
    def updateNoise(self):
        """
        Update each noise component. Then sum up the noise.
        """
        self.addComputePerDof("whiteNoise", "sqrt(1/dt) * gaussian")
        for i in range(self.nmodes):
            ## cosine xi-process will take in the white noise scaled by sqrt(kBT/tau/m/dt)
            self.addComputePerDof("NoiseCos{:d}".format(i),
                                  "NoiseCos{:d} + dt * whiteNoise".format(i))
            self.addComputePerDof("NoiseCos{:d}".format(i),
                                  "NoiseCos{:d} + dt * ( - NoiseCos{:d}/Taus{:d} - NoiseSin{:d}*Freqs{:d})".format(i,i,i,i,i))
            self.addComputePerDof("NoiseSin{:d}".format(i),
                                  "NoiseSin{:d} + dt * ( - NoiseSin{:d}/Taus{:d} + NoiseCos{:d}*Freqs{:d} )".format(i,i,i,i,i))
        ## sum up the noise
        self.addComputePerDof("Noise", "Noise*0")
        for i in range(self.nmodes):
            self.addComputePerDof("Noise", 
                                  "Noise + CoefNoiseCos{:d}*NoiseCos{:d} + CoefNoiseSin{:d}*NoiseSin{:d}".format(i,i,i,i))
                        
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
        self.addComputePerDof("v", "v + dt*Fv")
        self.addComputePerDof("v", "v + dt*Noise")
        ## update position
        self.addComputePerDof("x", "x + 0.5*dt*v")
        ## constraint positions
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")


if __name__ == '__main__':
    from sys import stdout
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas
    from openmm import *
    from openmm.app import *
    from openmm.unit import *
    from openmmtools.testsystems import IdealGas
    from gle_integrator import LangevinIntegrator, GLEIntegrator
    from utilities import Corr_t
    import h5py

    ## system
    nparticles = 32
    ndim = nparticles*3
    gas = IdealGas(nparticles=nparticles)
    temp = 298*kelvin
    timestep = 0.002 * picosecond
    kB=BOLTZMANN_CONSTANT_kB
    muc2 = 1.49241808560e-10 * openmm.unit.joule
    sl = 3.0e5 * openmm.unit.nanometer/openmm.unit.picosecond
    kbT = kB*temp / muc2 * openmm.unit.dalton * sl**2  # J-> dalton * (nm/ps)^2
    masses = np.array([gas.system.getParticleMass(i).value_in_unit(openmm.unit.dalton) for i in range(nparticles)]) * openmm.unit.dalton
    v2avg = kbT / masses
    noise_scale = (v2avg**0.5).value_in_unit(nanometer/picosecond)
    ## GLE parameters
    tau = 0.1 * picosecond
    max_period_in_tau = 4
    nbasis = 10
    max_period = tau * max_period_in_tau
    ws = 2*np.pi / max_period * np.arange(nbasis)
    ## load data
    mem_cf_cos = np.load("data/mem_cf_cos.npy")
    mem_cf_cos = - np.tile(mem_cf_cos, ndim).reshape(nparticles, 3, -1)
    mem_cf_sin = np.load("data/mem_cf_sin.npy")
    mem_cf_sin = - np.tile(mem_cf_sin, ndim).reshape(nparticles, 3, -1)
    noise_cf_cos = np.load("data/noise_cf_cos.npy") 
    noise_cf_cos = np.tile(noise_cf_cos, ndim).reshape(nparticles, 3, -1) * noise_scale[:,None,None]
    noise_cf_sin = np.load("data/noise_cf_sin.npy")
    noise_cf_sin = np.tile(noise_cf_sin, ndim).reshape(nparticles, 3, -1) * noise_scale[:,None,None]
    ## set up integrator
    integrator = GLEIntegrator(ws, mem_cf_cos, mem_cf_sin, 
                            ws, noise_cf_cos, noise_cf_sin, 
                            timestep, tau)
    ## simulation
    simulation = Simulation(gas.topology, gas.system, integrator)
    simulation.context.setPositions(gas.positions)

    simulation.reporters = []
    simulation.reporters.append(StateDataReporter(
        stdout, 1000, step=True, time=True,temperature=True, kineticEnergy=True, potentialEnergy=True,elapsedTime=True))
    simulation.reporters.append(StateDataReporter(
        "data/gas_le.csv",1000,time=True,potentialEnergy=True,kineticEnergy=True,totalEnergy=True,temperature=True,elapsedTime=True))
    simulation.step(20000)

    # h5_reporter = HDF5Reporter("data/gas_le.h5", 1, velocities=True)
    # simulation.reporters.append(h5_reporter)

    noise = []
    vel = []
    for i in range(20000):
        simulation.step(1)
        noise.append(integrator.getPerDofVariableByName('noise'))
        state = simulation.context.getState(getVelocities=True)
        vel.append(state.getVelocities().value_in_unit(nanometer/picosecond))

    noise = np.array(noise)
    vel = np.array(vel)
    np.save('data/noise.npy', noise)
    np.save('data/vel.npy', vel)