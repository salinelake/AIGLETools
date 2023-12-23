import numpy as np
import openmm
# from utilities import *



class LangevinIntegrator(openmm.CustomIntegrator):
    def __init__(self, timestep, temp, tau=1):
        super(LangevinIntegrator, self).__init__(timestep)
        ## constants
        muc2 = 1.49241808560e-10 * openmm.unit.joule
        sl = 3.0e5 * openmm.unit.nanometer/openmm.unit.picosecond
        kb = openmm.unit.BOLTZMANN_CONSTANT_kB
        ## to bypass the unit conversion issue in openmm8.0
        kbT = kb*temp / muc2 * openmm.unit.dalton * sl**2  # J-> dalton * (nm/ps)^2
        friction=1/tau
        self.addGlobalVariable("a", np.exp(-friction*timestep))
        self.addGlobalVariable("b", np.sqrt(1-np.exp(-2*friction*timestep)))
        self.addGlobalVariable("_kT", kbT)
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addConstrainVelocities()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + gaussian*b*sqrt(_kT/m)")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions();
        self.addComputePerDof("v", "v + (x-x1)/dt");

class GLEIntegrator(openmm.CustomIntegrator):
    """
    The integrator for generalized langevin equations (GLE).
    """
    def __init__(self, 
                 mem_omega:np.array, 
                 mem_cf_cos:np.array,
                 mem_cf_sin:np.array, 
                 noise_omega:np.array, 
                 noise_cf_cos:np.array,
                 noise_cf_sin:np.array,
                 timestep,  tau=1 ):
        """
        Args:
            mem_omega (np.array): shape=(d1), (value only)in unit of 1/picosecond
            mem_cf_cos (np.array): shape=(nparticle,3, d1), (value only) in unit of nm (1/ps)^2
            mem_cf_sin (np.array): shape=(nparticle,3, d1), (value only) in unit of (1/ps)^2
            noise_omega (np.array): shape=(d2), (value only)in unit of 1/picosecond
            noise_cf_cos (np.array): shape=(nparticle,3, d2), (value only) in unit of (1/ps)^2
            noise_cf_sin (np.array): shape=(nparticle,3, d2), (value only) in unit of (1/ps)^2
            timestep (float): in unit of picosecond
            tau (float): in unit of picosecond
        """
        super(GLEIntegrator, self).__init__(timestep)
        self.d1 = mem_omega.size
        self.d2 = noise_omega.size
 
        ## Define the integrator
        ## add constants and variables
        self.addGlobalVariable('tau', tau)
        self.addPerDofVariable('noise',0)
        self.addPerDofVariable('Fv',0)
        self.addPerDofVariable('whiteNoise',0)
        self.addMemoryComponents(mem_omega, mem_cf_cos, mem_cf_sin)
        self.addNoiseComponents(noise_omega, noise_cf_cos, noise_cf_sin)
        self.addPerDofVariable("x1", 0)
        ## add integration scheme
        self.setIntegrationScheme()
        

    def addMemoryComponents(self, mem_omega, mem_cf_cos, mem_cf_sin):
        """
        Adds memory components to the integrator.
        FvCos[i] and Fvsin[i] value only, in the unit of length (nm).
        CoefFvcos[i] and CoefFvsin[i] value only, in the the unit of (1/ps)^2.
        FvFreq[i] has the unit of (1/ps)
        """
        for i in range(self.d1):
            ## frequency of the memory
            self.addGlobalVariable('FvFreq{:d}'.format(i), mem_omega[i])
            ## the memory components
            self.addPerDofVariable('FvCos{:d}'.format(i), 0)
            self.addPerDofVariable('FvSin{:d}'.format(i), 0)
            ## the weight of the memory components in Fv
            self.addPerDofVariable('CoefFvCos{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefFvCos{:d}'.format(i), mem_cf_cos[...,i] )
            self.addPerDofVariable('CoefFvSin{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefFvSin{:d}'.format(i), mem_cf_sin[...,i])
    
    def addNoiseComponents(self, noise_omega, noise_cf_cos, noise_cf_sin):
        """
        Adds noise components to the integrator.
        XiCos[i] and XiSin[i] have the unit of length (nm).
        CoefXiCos[i] and CoefXiSin[i] have the unit of (1/ps)^2.
        """
        for i in range(self.d2):
            ## frequency of the xi process
            self.addGlobalVariable('XiFreq{:d}'.format(i), noise_omega[i])
            ## the xi process
            self.addPerDofVariable('XiCos{:d}'.format(i),0)
            self.addPerDofVariable('XiSin{:d}'.format(i),0)
            ## the weight of the xi process in noise
            self.addPerDofVariable('CoefXiCos{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefXiCos{:d}'.format(i), noise_cf_cos[...,i] )
            self.addPerDofVariable('CoefXiSin{:d}'.format(i), 0 )
            self.setPerDofVariableByName('CoefXiSin{:d}'.format(i), noise_cf_sin[...,i])
        
    def sumFv(self):
        """
        Update the total memory force Fv
        """
        self.addComputePerDof("Fv", "Fv*0")
        for i in range(self.d1):
            self.addComputePerDof("Fv", 
                                  "Fv + CoefFvCos{:d}*FvCos{:d} + CoefFvSin{:d}*FvSin{:d}".format(i,i,i,i))
    
    def sumNoise(self):
        """
        Update the total noise
        """
        self.addComputePerDof("noise", "noise*0")
        for i in range(self.d2):
            self.addComputePerDof("noise", 
                                  "noise + CoefXiCos{:d}*XiCos{:d} + CoefXiSin{:d}*XiSin{:d}".format(i,i,i,i))
            
    def updateMemory(self):
        """
        Update each memory component
        """
        for i in range(self.d1):
            self.addComputePerDof("FvCos{:d}".format(i), 
                                  "FvCos{:d} + dt*v".format(i))
            self.addComputePerDof("FvCos{:d}".format(i), 
                                  "FvCos{:d} + dt*( - FvCos{:d}/tau - FvSin{:d}*FvFreq{:d})".format(i,i,i,i))
            self.addComputePerDof("FvSin{:d}".format(i),
                                  "FvSin{:d} + dt*( - FvSin{:d}/tau + FvCos{:d}*FvFreq{:d})".format(i,i,i,i))
    
    def updateNoise(self):
        """
        Update each xi process
        """
        self.addComputePerDof("whiteNoise", "sqrt(1/dt) * gaussian")
        for i in range(self.d2):
            ## cosine xi-process will take in the white noise scaled by sqrt(kBT/tau/m/dt)
            self.addComputePerDof("XiCos{:d}".format(i),
                                  "XiCos{:d} + dt * whiteNoise".format(i))
            self.addComputePerDof("XiCos{:d}".format(i),
                                  "XiCos{:d} + dt * ( - XiCos{:d}/tau - XiSin{:d}*XiFreq{:d})".format(i,i,i,i))
            self.addComputePerDof("XiSin{:d}".format(i),
                                  "XiSin{:d} + dt * ( - XiSin{:d}/tau + XiCos{:d}*XiFreq{:d} )".format(i,i,i,i))
            
    def setIntegrationScheme(self):
        """
        scheme: ABOBA -> BOBAA
        B: x->x+vdt/2
        O: v->v+(F/m+Fv+noise)dt
        AA: update Fv and noise by dt
        """
        self.addUpdateContextState()
        self.addConstrainVelocities()
        ## AA
        self.updateMemory()
        self.updateNoise()
        ## B
        self.addComputePerDof("x", "x + 0.5*dt*v")
        ## O
        self.sumNoise()
        self.sumFv()
        self.addComputePerDof("v", "v + dt*f/m")
        self.addComputePerDof("v", "v + dt*Fv")
        self.addComputePerDof("v", "v + dt*noise")
        ## B
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