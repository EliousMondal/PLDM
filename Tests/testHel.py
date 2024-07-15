import numpy as np
import parameters as param
from numba import jit


eyeN = np.identity(param.NStates)
# R =  np.random.random(param.NR) * 1000


@jit(nopython=True)
def H_SB(R):
    """ 
        description → Calculating the system bath coupling
        input       → bath coordinates R (NR)
        output      → system bath coupling, ˢᵇεⱼ= ∑ᵢ(cᵢRᵢʲ)
    """
    cR = (param.cj_nr * R).reshape(param.NMol, param.NModes)
    return np.sum(cR, axis=1)



@jit(nopython=True)
def Hel_Chebyshev(R):
    '''Electronic diabatic Hamiltonian'''
    Vij = np.zeros((param.NStates, param.NStates), dtype=np.complex128)

    for i in range(param.NMol):
        Vij[i+1, i+1] = np.sum(param.cj * R[i * param.NModes : (i+1) * param.NModes]) + param.δε[i]
        Vij[i+1, -1], Vij[-1, i+1] = param.g, param.g         

    Vij = (2 / param.ΔE) * (Vij - (param.E_min * eyeN)) - eyeN
    Vij[0, 0] = 0.0

    return Vij



@jit(nopython=True)
def Hψ(ψ, ε_SB):
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
    
    εi = (2 / param.ΔE) * (param.δε + ε_SB - param.E_min) - 1.0
    ψt[1: param.NMol+1]  = ψ[1: param.NMol+1] * (εi).astype(np.complex128)
    ψt[param.NStates-1]  = param.Δ * ψ[param.NStates-1]
    
    ψt[1: param.NMol+1] += param.g_norm * ψ[param.NStates-1]
    ψt[param.NStates-1] += param.g_norm * np.sum(ψ[1: param.NMol+1])
    ψt[0] = 0.0
    
    return ψt


# ψtest = np.random.uniform(-1, 1, param.NStates) + 1j * np.random.uniform(-1, 1, param.NStates)
# ψtest /= np.linalg.norm(ψtest)

# ψtest_new = ψtest.copy()
# ψtest_old = ψtest.copy()

# ε_SB = H_SB(R)

# ψ_old = Hel_Chebyshev(R) @ ψtest_old
# ψ_new = Hψ(ψtest_new, ε_SB)

# print("Real part")
# print("ψ_old      = ", np.round(ψ_old.real, 8))
# print("ψ_new      = ", np.round(ψ_new.real, 8))
# print("difference = ", np.round((ψ_old - ψ_new).real, 8))
# print("\n")

# print("Imag part")
# print("ψ_old      = ", np.round(ψ_old.imag, 8))
# print("ψ_new      = ", np.round(ψ_new.imag, 8))
# print("difference = ", np.round((ψ_old - ψ_new).imag, 8))
# print("\n")