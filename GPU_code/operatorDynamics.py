import numpy as np
import torch
from mpi4py import MPI
import time
import sys
# import os  
# os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
# os.environ['TORCH_USE_CUDA_DSA']
# os.environ['CUDA_LAUNCH_BLOCKING=1']  

import pldm as method
import parameters as param
import model
import trajClass as tc


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TrajFolder = sys.argv[1]
NTraj = 20

NTasks = NTraj // size
NRem = NTraj - (NTasks*size)
TaskArray = [i for i in range(rank * NTasks , (rank+1) * NTasks)]
for i in range(NRem):
    if i == rank: 
        TaskArray.append((NTasks*size)+i)
print(TaskArray)
print(f"Rank {rank} has {len(TaskArray)} number of trajectories")

print(torch.cuda.is_available())
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu_device = torch.device("meta")
print(gpu_device)

# dummy variables for compiling the functions 
stCompile  = time.time()
dum_mols   = 2
dum_steps  = 3
dum_cols   = 4
dumTraj    = tc.trajData(dum_mols, dum_steps, dum_cols, gpu_device)
print("dummpy traj created", flush=True)

# compiling model functions
print(f"Compiling model functions for process {rank}")
model.initMapping(dumTraj)
model.H_SB(dumTraj)
model.Hψ(dumTraj.ψ, dumTraj)
# model.μψ(dumTraj.ψ, dumTraj)

# compiling dynamics functions
print(f"Compiling dynamics functions for process {rank}")
method.Force(dumTraj, dumTraj.F1)
method.Umap(dumTraj)
method.VelVer(dumTraj)
method.runTraj(dumTraj)

edCompile = time.time()
print(f"All model and dynamics functions compiled in {np.round(edCompile - stCompile, 3)} seconds for process {rank}.\n")


# simulation variables
# operators       = torch.tensor([[0, i] for i in range(1, param.NMol+1)], dtype=int, device=gpu_device)    
operators1      = torch.tensor([[0, i] for i in range(1, param.NMol+1) ], dtype=int, device=gpu_device)
operators2      = torch.tensor([[i, 0] for i in range(1, param.NMol+1) ], dtype=int, device=gpu_device)
operators       = torch.vstack((operators1, operators2))
# print(operators1.shape, operators2.shape, operators.shape)
# exit()

trajData        = tc.trajData(param.NMol, param.NSteps, operators.shape[0], gpu_device)
trajData.δε     = torch.zeros((param.NMol, operators.shape[0]), device=gpu_device)                 # param.δε

for itraj in TaskArray: 
    
    trajData.iF[:]  = operators[:, 0]
    trajData.iB[:]  = operators[:, 1]
    
    # print(f"Loading trajectory data ...", flush=True)
    # itrajR = np.loadtxt(TrajFolder + f"{itraj+1}/initial_R_{itraj+1}.txt")
    # itrajP = np.loadtxt(TrajFolder + f"{itraj+1}/initial_P_{itraj+1}.txt")   
    itrajR = np.loadtxt(TrajFolder + f"{1}/initial_R_{1}.txt")
    itrajP = np.loadtxt(TrajFolder + f"{1}/initial_P_{1}.txt") 
    
    R_data  = torch.broadcast_to(torch.from_numpy(itrajR), (trajData.NCols, trajData.NR)).T
    P_data  = torch.broadcast_to(torch.from_numpy(itrajP), (trajData.NCols, trajData.NR)).T
    trajData.R[:] = R_data.to(gpu_device)
    trajData.P[:] = P_data.to(gpu_device)
    # print(f"Trajectory data loaded successfully for process {rank}.", flush=True)
    
    st = time.time()
    method.runTraj(trajData)
    ed = time.time()
    print(f"jobs for rank {rank} finished in {ed-st} seconds", flush=True)

    # iCol: np.int32
    # qFt_cpu = torch.Tensor.cpu(trajData.qFt)
    # qBt_cpu = torch.Tensor.cpu(trajData.qBt)
    # pFt_cpu = torch.Tensor.cpu(trajData.pFt)
    # pBt_cpu = torch.Tensor.cpu(trajData.pBt)
    
    # for iCol in range(trajData.NCols):
    #     np.savetxt(TrajFolder + f"{itraj+1}/qFt_{itraj+1}_{trajData.iF[iCol]}{trajData.iB[iCol]}.txt", torch.Tensor.numpy(qFt_cpu[:, :, iCol]), fmt="% 24.18e")
    #     np.savetxt(TrajFolder + f"{itraj+1}/qBt_{itraj+1}_{trajData.iF[iCol]}{trajData.iB[iCol]}.txt", torch.Tensor.numpy(qBt_cpu[:, :, iCol]), fmt="% 24.18e")
    #     np.savetxt(TrajFolder + f"{itraj+1}/pFt_{itraj+1}_{trajData.iF[iCol]}{trajData.iB[iCol]}.txt", torch.Tensor.numpy(pFt_cpu[:, :, iCol]), fmt="% 24.18e")
    #     np.savetxt(TrajFolder + f"{itraj+1}/pBt_{itraj+1}_{trajData.iF[iCol]}{trajData.iB[iCol]}.txt", torch.Tensor.numpy(pBt_cpu[:, :, iCol]), fmt="% 24.18e")