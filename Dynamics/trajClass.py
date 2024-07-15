import numpy as np
from numba import int32, float64, complex128
from numba.experimental import jitclass
from numba import jit

import parameters as param


spec = [
    ('nmols',            int32),   # number of molecules
    ('nsteps',           int32),   # number of time steps
    ('nstates',          int32),   # number of states
    ('R',           float64[:]),   # position of bath modes
    ('P',           float64[:]),   # momentum of bath modes
    ('v',           float64[:]),   # velocity of bath modes
    ('cj_nr',       float64[:]),   # bath coupling coefficient
    ('ωj_nr_sq',    float64[:]),   # kinetic energy of
    ('δε',          float64[:]),   # energy inhomogenity
    ('ε_SB',        float64[:]),   # bath fluctuations
    ('ψ',        complex128[:]),   # forward or backward wavefunction
    ('qF',          float64[:]),   # position of forward wavefunction
    ('qB',          float64[:]),   # position of backward wavefunction
    ('pF',          float64[:]),   # momentum expansion of forward wavefunction
    ('pB',          float64[:]),   # momentum expansion of backward wavefunction
    ('ρii',         float64[:]),   # population of state i
    ('iF',               int32),   # forward mode index
    ('iB',               int32),   # backward mode index
    ('F1',          float64[:]),   # force
    ('F2',          float64[:]),   # force 
    ('ψF',       complex128[:]),   # forward wavefunction
    ('ψF1',      complex128[:]),   # intermediate forward wavefunction
    ('ψF2',      complex128[:]),   # intermediate forward wavefunction
    ('ψF3',      complex128[:]),   # intermediate forward wavefunction
    ('ψFt',      complex128[:]),   # intermediate forward wavefunction
    ('ψFΔ',      complex128[:]),   # final forward wavefunction
    ('ψB',       complex128[:]),   # backward wavefunction
    ('ψB1',      complex128[:]),   # intermediate backward wavefunction
    ('ψB2',      complex128[:]),   # intermediate backward wavefunction
    ('ψB3',      complex128[:]),   # intermediate backward wavefunction
    ('ψBt',      complex128[:]),   # intermediate backward wavefunction
    ('ψBΔ',      complex128[:]),   # final backward wavefunction
    ('qFt',      float64[:, :]),   # time dependent qF
    ('qBt',      float64[:, :]),   # time dependent qB
    ('pFt',      float64[:, :]),   # time dependent pF
    ('pBt',      float64[:, :])    # time dependent pB
]


@jitclass(spec)
class trajData(object):
    def __init__(self, nmols, nsteps):
        self.nmols    = nmols
        self.nsteps   = nsteps
        self.nstates  = self.nmols + 2
        
        self.R        = np.zeros(param.NModes * self.nmols)
        self.P        = np.zeros_like(self.R)
        self.v        = np.zeros_like(self.P)
        
        self.cj_nr    = np.kron(np.ones(self.nmols), param.cj)
        self.ωj_nr_sq = np.kron(np.ones(self.nmols), param.ωj) ** 2
        
        self.δε       = np.zeros(self.nmols)
        self.ε_SB     = np.zeros_like(self.δε)
        
        self.ψ        = np.zeros(self.nstates, dtype=np.complex128)
        
        self.qF       = np.zeros(self.nstates)
        self.qB       = np.zeros_like(self.qF)
        self.pF       = np.zeros_like(self.qF)
        self.pB       = np.zeros_like(self.qF)
        
        self.ρii      = np.zeros(self.nmols)
        
        self.iF       = 0
        self.iB       = 0
        
        self.F1       = np.zeros_like(self.R)
        self.F2       = np.zeros_like(self.R)
        
        self.ψF       = np.zeros_like(self.ψ)
        self.ψF1      = np.zeros_like(self.ψ)
        self.ψF2      = np.zeros_like(self.ψ)
        self.ψF3      = np.zeros_like(self.ψ)
        self.ψFt      = np.zeros(self.nstates-1, dtype=np.complex128)
        self.ψFΔ      = np.zeros_like(self.ψ)
        
        self.ψB       = np.zeros_like(self.ψ)
        self.ψB1      = np.zeros_like(self.ψ)
        self.ψB2      = np.zeros_like(self.ψ)
        self.ψB3      = np.zeros_like(self.ψ)
        self.ψBt      = np.zeros_like(self.ψFt)
        self.ψBΔ      = np.zeros_like(self.ψ)
        
        self.qFt      = np.zeros((self.nsteps, self.nstates))
        self.qBt      = np.zeros_like(self.qFt)
        self.pFt      = np.zeros_like(self.qFt)
        self.pBt      = np.zeros_like(self.qFt)