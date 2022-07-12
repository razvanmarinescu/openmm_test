from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import time
start = time.time()
print('Simulating a system of 4 proteins (dimer of dimers): https://www.rcsb.org/structure/2g33')

GPU_DEVICE=1 # which GPUs to run on 

pdb = PDBFile('bstate.pdb')
psf = CharmmPsfFile('bstate.psf')

params = CharmmParameterSet('toppar_c36_jul21/top_all36_prot.rtf', 'toppar_c36_jul21/par_all36m_prot.prm')

system = psf.createSystem(params, nonbondedMethod=NoCutoff, nonbondedCutoff=1*nanometer, constraints=HBonds, implicitSolvent=GBn2)

# Langevin Middle Integrator is more accurate for larger timesteps
# 4fs ok, but need to keep constraints on HBonds (I think)
# 5fs works, but need heavy hydrogens (mass redistribution from heavy atoms)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '%s' % GPU_DEVICE} # implicit solvent doesn't support more than 1 GPU
simulation = Simulation(psf.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
print('Number of atoms:', len(pdb.positions))

simulation.loadState('parent.xml')
simulation.reporters.append(DCDReporter('seg.dcd', 1000)) 
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, speed=True))
simulation.step(25000)

end = time.time()
print('elapsed seconds:', end - start)
