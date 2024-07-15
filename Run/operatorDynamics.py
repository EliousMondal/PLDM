import numpy as np
from mpi4py import MPI
import time
import sys

import pldm as method
import parameters as param
import model
import trajClass as tc


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TrajFolder = sys.argv[1]
NTraj = param.NTraj

NTasks = NTraj//size
NRem = NTraj - (NTasks*size)
TaskArray = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)

print(f"Rank {rank} has {len(TaskArray)} number of trajectories")


# dummy variables for compiling the functions 
stCompile  = time.time()
dum_mols   = 2
dum_steps  = 3
dumTraj    = tc.trajData(dum_mols, dum_steps)

# compiling model functions
print(f"Compiling model functions for process {rank}")
model.initMapping(dumTraj)
model.H_SB(dumTraj)
model.Hψ(dumTraj.ψ, dumTraj)
model.μψ(dumTraj.ψ, dumTraj)

# compiling dynamics functions
print(f"Compiling dynamics functions for process {rank}")
method.Force(dumTraj, dumTraj.F1)
method.Umap(dumTraj)
method.VelVer(dumTraj)
method.runTraj(dumTraj)

edCompile = time.time()
print(f"All model and dynamics functions compiled in {np.round(edCompile - stCompile, 3)} seconds for process {rank}.\n")


# simulation variables
trajData = tc.trajData(param.NMol, param.NSteps)
trajData.δε = param.δε

# dipole test
operators = np.array([[0, i] for i in range(1, param.NMol+1)], dtype=int)


st = MPI.Wtime()
for itraj in TaskArray: 
    itrajR = np.loadtxt(TrajFolder + f"{itraj+1}/initial_R_{itraj+1}.txt")
    itrajP = np.loadtxt(TrajFolder + f"{itraj+1}/initial_P_{itraj+1}.txt")   
    for op in operators:
        iF, iB = op[0], op[1]

        trajData.iF = iF
        trajData.iB = iB
        trajData.R[:] = itrajR[:]
        trajData.P[:] = itrajP[:]
        
        method.runTraj(trajData)
        np.savetxt(TrajFolder + f"{itraj+1}/qFt_{itraj+1}_{iF}{iB}.txt", trajData.qFt, fmt="% 24.18e")
        np.savetxt(TrajFolder + f"{itraj+1}/qBt_{itraj+1}_{iF}{iB}.txt", trajData.qBt, fmt="% 24.18e")
        np.savetxt(TrajFolder + f"{itraj+1}/pFt_{itraj+1}_{iF}{iB}.txt", trajData.pFt, fmt="% 24.18e")
        np.savetxt(TrajFolder + f"{itraj+1}/pBt_{itraj+1}_{iF}{iB}.txt", trajData.pBt, fmt="% 24.18e")

ed = MPI.Wtime()
print(f"jobs for rank {rank} finished in {ed-st} seconds")