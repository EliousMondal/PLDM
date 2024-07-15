import numpy as np
from numba import jit

import parameters as param


@jit(nopython=True)
def H_SB(data):
    """ 
        description → Calculating the system bath coupling
        input       → bath coordinates R (NR)
        output      → system bath coupling, ˢᵇεⱼ= ∑ᵢ(cᵢRᵢʲ)
    """
    cR = (data.cj_nr * data.R).reshape(data.nmols, param.NModes)
    data.ε_SB[:] = np.sum(cR, axis=1)


@jit(nopython=True)
def Hψ(ψ, data):
    """
        description → the action of normalised Hψ (according to chebyshev method)
            Hψ = ∑ᵢ(εᵢψᵢ + g)|Eᵢ⁰⟩ + (ω+∑ᵢgψᵢ)|G¹⟩
        input:
            1) ψ    → the wavefunction at current time
            2) ε_SB → the system-bath interaction energy
        output      → Hₙψ
            where Hₙ = (2 / ΔE)(H - E_min) - 1
    """
    
    ψt = np.zeros_like(ψ)
    nstates = data.nmols + 2
    
    εi = (2 / param.ΔE) * (data.δε + data.ε_SB - param.E_min) - 1.0
    ψt[1: data.nmols+1]  = ψ[1: data.nmols+1] * (εi).astype(np.complex128)
    ψt[nstates-1]  = param.Δ * ψ[nstates-1]
    
    ψt[1: data.nmols+1] += param.g_norm * ψ[nstates-1]
    ψt[nstates-1] += param.g_norm * np.sum(ψ[1: data.nmols+1])
    ψt[0] = 0.0
    
    return ψt
    
    
@jit(nopython=True)
def μψ(ψ, data):
    """
        description → the action of μψ
            Hψ = ψ[0]∑ᵢμᵢ|Eᵢ⁰⟩ + (∑ᵢμᵢψᵢ)|G¹⟩ (μᵢ dependence not added yet)
        input: ψ    → the wavefunction at current time
        output      → μψ
    """
    δψ = np.zeros_like(ψ)
    δψ[0] = np.sum(ψ[1: data.nmols+1])
    δψ[1: data.nmols+1] = ψ[0]
    
    return δψ


@jit(nopython=True)
def initMapping(data):
    
    data.qF[:] = 0.0
    data.qB[:] = 0.0
    data.pF[:] = 0.0
    data.pB[:] = 0.0
    
    data.qF[data.iF], data.qB[data.iB] = 1.0,  1.0 
    data.pF[data.iF], data.pB[data.iB] = 1.0, -1.0
