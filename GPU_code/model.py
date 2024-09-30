import numpy as np
import torch
import parameters as param


def H_SB(data):
    """ 
        description → Calculating the system bath coupling
        input       → bath coordinates R (NR)
        output      → system bath coupling, ˢᵇεⱼ= ∑ᵢ(cᵢRᵢʲ)
    """

    cR = torch.einsum('i, ij -> ij', data.cj_nr, data.R)
    data.ε_SB[:, :] = torch.sum(cR.reshape(data.nmols, param.NModes, data.NCols), axis=1)


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
    
    ψt = torch.zeros_like(ψ)
    nstates = data.nmols + 2
    
    εi = (2 / param.ΔE) * (data.δε + data.ε_SB - param.E_min) - 1.0
    ψt[1: data.nmols+1, :]  = ψ[1: data.nmols+1, :] * εi #.astype(torch.complex128)
    ψt[nstates-1, :]  = param.Δ * ψ[nstates-1, :]
    
    ψt[1: data.nmols+1, :] += param.g_norm * ψ[nstates-1, :]
    ψt[nstates-1, :] += param.g_norm * torch.sum(ψ[1: data.nmols+1, :], axis=0)
    ψt[0, :] *= 0.0
    
    return ψt
    
  
# def μψ(ψ, data):
#     """
#         description → the action of μψ
#             Hψ = ψ[0]∑ᵢμᵢ|Eᵢ⁰⟩ + (∑ᵢμᵢψᵢ)|G¹⟩ (μᵢ dependence not added yet)
#         input: ψ    → the wavefunction at current time
#         output      → μψ
#     """

#     # δψ = torch.zeros_like(ψ)
#     data.δψ[0, :] = torch.sum(ψ[1: data.nmols+1, :], axis=0)
#     data.δψ[1: data.nmols+1, :] = torch.broadcast_to(ψ[0, :], (data.nmols, data.NCols))
    
#     # return δψ


def initMapping(data):
    
    data.qF[:, :] *= 0.0
    data.qB[:, :] *= 0.0
    data.pF[:, :] *= 0.0
    data.pB[:, :] *= 0.0
    
    for iCol in range(data.NCols):
        data.qF[data.iF[iCol], iCol] =  1.0 
        data.qB[data.iB[iCol], iCol] =  1.0 
        data.pF[data.iF[iCol], iCol] =  1.0 
        data.pB[data.iB[iCol], iCol] = -1.0
