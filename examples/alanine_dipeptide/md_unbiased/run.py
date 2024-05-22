from sys import stdout
import mdtraj
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

pdb = PDBFile("init3.pdb")
forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, constraints=HBonds)

## NPT parameters
temperature = 300 * kelvin
pressure = 1 * bar
dt = 2 * femtoseconds
tau_t = 10 * picosecond

## MD setup
integrator = LangevinIntegrator(temperature, 1 / tau_t, dt)
system.addForce(MonteCarloBarostat(pressure, temperature))
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

## trajectory
ad = list(list(pdb.topology.chains())[0].atoms())
subset_indices = [atom.index for atom in ad]
traj_reporter= mdtraj.reporters.DCDReporter("./traj_100ns.dcd", 2, atomSubset=subset_indices)
simulation.reporters.append(traj_reporter)

## printing
simulation.reporters.append(
    StateDataReporter(stdout, 50000, step=True, time=True, temperature=True, elapsedTime=True)
)

## logging
simulation.reporters.append(
    StateDataReporter(
        "./traj_100ns.csv",
        50000,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        elapsedTime=True
    )
)

## run
simulation.step(50000000)