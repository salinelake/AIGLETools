from sys import stdout

import mdtraj
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
import openmm.app.metadynamics as mtd
from openmm.openmm import CustomTorsionForce, CustomCVForce

from utilities import *
pdb = PDBFile("init3.pdb")
forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

# The modeller builds a periodic box with the solute and solvent molecules.
# PME is the method to compute long-range electristatic interactions in
# periodic systems.
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)

## NPT parameters
temperature = 300 * kelvin
pressure = 1 * bar
dt = 2 * femtoseconds
tau_t = picosecond

## MD setup
integrator = LangevinIntegrator(temperature, 1 / tau_t, dt)
system.addForce(MonteCarloBarostat(pressure, temperature))
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

## trajectory
# simulation.reporters.append(DCDReporter("traj3.dcd", 500))
ad = list(list(pdb.topology.chains())[0].atoms())
subset_indices = [atom.index for atom in ad]
traj_reporter= mdtraj.reporters.DCDReporter("metad.dcd", 100, atomSubset=subset_indices)
simulation.reporters.append(traj_reporter)

## printing
simulation.reporters.append(
    StateDataReporter(stdout, 200, step=True, time=True, temperature=True, elapsedTime=True)
)
simulation.step(1000)

## get torsion index
traj = mdtraj.load("metad.dcd", top="alanine-dipeptide.pdb")
# plot_ramachandran(traj)
phi_idx, phi = mdtraj.compute_phi(traj)
psi_idx, psi = mdtraj.compute_psi(traj)

## define "collective variable" as energy function
force_phi = CustomTorsionForce('theta')
force_phi.addTorsion(phi_idx[0][0],phi_idx[0][1],phi_idx[0][2],phi_idx[0][3])
force_psi = CustomTorsionForce('theta')
force_psi.addTorsion(psi_idx[0][0],psi_idx[0][1],psi_idx[0][2],psi_idx[0][3])

## define collective variable
cv_phi = CustomCVForce('phi')
cv_phi.addCollectiveVariable('phi', force_phi)
cv_psi = CustomCVForce('psi')
cv_psi.addCollectiveVariable('psi', force_psi)

bv_phi = mtd.BiasVariable(cv_phi, minValue=-np.pi, maxValue=np.pi, biasWidth=5/180*np.pi, periodic=True, gridWidth=None)
bv_psi = mtd.BiasVariable(cv_psi, minValue=-np.pi, maxValue=np.pi, biasWidth=5/180*np.pi, periodic=True, gridWidth=None)

# Set up the context for mtd simulation
# at this step the CV and the system are separately passed to Metadynamics
metad = mtd.Metadynamics(system, 
                        [bv_phi,bv_psi], 
                        temperature, 
                        biasFactor = 10.0, 
                        height = 1.0 * kilojoules_per_mole, 
                        frequency = 500, 
                        saveFrequency = 500, 
                        biasDir = './biases')

## setup simulation
positions = simulation.context.getState(getPositions=True).getPositions()
simulation.context.reinitialize()
integrator = LangevinIntegrator(temperature, 1 / tau_t, dt)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(positions)
## setup reporters
traj_reporter= mdtraj.reporters.DCDReporter("metad.dcd", 5000, atomSubset=subset_indices)
simulation.reporters.append(traj_reporter)
simulation.reporters.append(StateDataReporter(stdout, 5000, 
        step=True, time=True, temperature=True, elapsedTime=True))
simulation.reporters.append(StateDataReporter("metad.csv", 5000, 
        time=True, potentialEnergy=True, totalEnergy=True, temperature=True,))

# Run small-scale simulation and plot the free energy landscape
metad.step(simulation, 10000000)
np.save( './biases/free_energy.npy', metad.getFreeEnergy())