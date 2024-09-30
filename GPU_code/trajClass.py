import numpy as np
import torch
# from numba import int32, float64, complex128
# from numba.experimental import jitclass
# from numba import jit

import parameters as param


# spec = [
#     ('nmols',            int32),   # number of molecules
#     ('nsteps',           int32),   # number of time steps
#     ('nstates',          int32),   # number of states
#     ('R',           float64[:]),   # position of bath modes
#     ('P',           float64[:]),   # momentum of bath modes
#     ('v',           float64[:]),   # velocity of bath modes
#     ('cj_nr',       float64[:]),   # bath coupling coefficient
#     ('ωj_nr_sq',    float64[:]),   # kinetic energy of
#     ('δε',          float64[:]),   # energy inhomogenity
#     ('ε_SB',        float64[:]),   # bath fluctuations
#     ('ψ',        complex128[:]),   # forward or backward wavefunction
#     ('qF',          float64[:]),   # position of forward wavefunction
#     ('qB',          float64[:]),   # position of backward wavefunction
#     ('pF',          float64[:]),   # momentum expansion of forward wavefunction
#     ('pB',          float64[:]),   # momentum expansion of backward wavefunction
#     ('ρii',         float64[:]),   # population of state i
#     ('iF',               int32),   # forward mode index
#     ('iB',               int32),   # backward mode index
#     ('F1',          float64[:]),   # force
#     ('F2',          float64[:]),   # force 
#     ('ψF',       complex128[:]),   # forward wavefunction
#     ('ψF1',      complex128[:]),   # intermediate forward wavefunction
#     ('ψF2',      complex128[:]),   # intermediate forward wavefunction
#     ('ψF3',      complex128[:]),   # intermediate forward wavefunction
#     ('ψFt',      complex128[:]),   # intermediate forward wavefunction
#     ('ψFΔ',      complex128[:]),   # final forward wavefunction
#     ('ψB',       complex128[:]),   # backward wavefunction
#     ('ψB1',      complex128[:]),   # intermediate backward wavefunction
#     ('ψB2',      complex128[:]),   # intermediate backward wavefunction
#     ('ψB3',      complex128[:]),   # intermediate backward wavefunction
#     ('ψBt',      complex128[:]),   # intermediate backward wavefunction
#     ('ψBΔ',      complex128[:]),   # final backward wavefunction
#     ('qFt',      float64[:, :]),   # time dependent qF
#     ('qBt',      float64[:, :]),   # time dependent qB
#     ('pFt',      float64[:, :]),   # time dependent pF
#     ('pBt',      float64[:, :])    # time dependent pB
# ]


# @jitclass(spec)
class trajData(object):
    def __init__(self, nmols, nsteps, NCols, gd):
        self.nmols    = nmols
        self.NR       = param.NModes * self.nmols
        self.nsteps   = nsteps
        self.nstates  = self.nmols + 2
        self.NCols    = NCols
        self.gdevice  = gd
        
        self.R        = torch.zeros((self.NR, self.NCols), device = self.gdevice)
        self.P        = torch.zeros_like(self.R, device = self.gdevice)
        self.v        = torch.zeros_like(self.P, device = self.gdevice)
        
        self.cj       = torch.tensor(param.cj, device = self.gdevice)
        self.ωj       = torch.tensor(param.ωj, device = self.gdevice)
        self.ones     = torch.ones(self.nmols, device = self.gdevice)
        self.cj_nr    = torch.kron(self.ones, self.cj)
        self.cj_nr.to(self.gdevice)
        self.ωj_nr_sq = torch.kron(self.ones, self.ωj ** 2)
        self.ωj_nr_sq.to(self.gdevice)
        
        self.δε       = torch.zeros((self.nmols, self.NCols), device = self.gdevice)
        self.ε_SB     = torch.zeros_like(self.δε, device = self.gdevice)
        
        self.ψ        = torch.zeros((self.nstates, self.NCols), dtype=torch.complex128, device = self.gdevice)
        self.ψt       = torch.zeros_like(self.ψ, device = self.gdevice)
        self.δψ       = torch.zeros_like(self.ψ, device = self.gdevice)
        
        self.qF       = torch.zeros((self.nstates, self.NCols), device = self.gdevice)
        self.qB       = torch.zeros_like(self.qF, device = self.gdevice)
        self.pF       = torch.zeros_like(self.qF, device = self.gdevice)
        self.pB       = torch.zeros_like(self.qF, device = self.gdevice)
        
        self.ρii      = torch.zeros((self.nmols, self.NCols), device = self.gdevice)
        
        self.iF       = torch.zeros((self.NCols), dtype=torch.int32, device = self.gdevice)
        self.iB       = torch.zeros((self.NCols), dtype=torch.int32, device = self.gdevice)
        
        self.F1       = torch.zeros_like(self.R, device = self.gdevice)
        self.F2       = torch.zeros_like(self.R, device = self.gdevice)
        
        self.ψF       = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψF1      = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψF2      = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψF3      = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψFt      = torch.zeros((self.nstates-1, self.NCols), dtype=torch.complex128, device = self.gdevice)
        self.ψFΔ      = torch.zeros_like(self.ψ, device = self.gdevice)
        
        self.ψB       = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψB1      = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψB2      = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψB3      = torch.zeros_like(self.ψ, device = self.gdevice)
        self.ψBt      = torch.zeros_like(self.ψFt, device = self.gdevice)
        self.ψBΔ      = torch.zeros_like(self.ψ, device = self.gdevice)
        
        self.qFt      = torch.zeros((self.nsteps, self.nstates, self.NCols), device = self.gdevice)
        self.qBt      = torch.zeros_like(self.qFt, device = self.gdevice)
        self.pFt      = torch.zeros_like(self.qFt, device = self.gdevice)
        self.pBt      = torch.zeros_like(self.qFt, device = self.gdevice)
        
        
        
if __name__ == "__main__":
    dum_mols   = 2
    dum_steps  = 3
    dum_cols   = 4
    dumTraj    = trajData(dum_mols, dum_steps, dum_cols)
    print("dummpy traj created", flush=True)