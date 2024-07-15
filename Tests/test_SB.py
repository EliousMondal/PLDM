import numpy as np
import parameters as param
from numba import jit


@jit(nopython=True)
def H_SB_old(R):
    εSB = np.zeros(param.NMol)
    for i in range(param.NMol):
        εSB[i] = np.sum(param.cj * R[i * param.NModes : (i+1) * param.NModes])
    return εSB
    

@jit(nopython=True)
def H_SB(R):
    """ 
        description → Calculating the system bath coupling
        input       → bath coordinates R (NR)
        output      → system bath coupling, ˢᵇεⱼ= ∑ᵢ(cᵢRᵢʲ)
    """
    cR = (param.cj_nr * R).reshape(param.NMol, param.NModes)
    return np.sum(cR, axis=1)


Rtest = np.random.random(param.NR) * 1000

εSB_old = H_SB_old(Rtest)
εSB_new = H_SB(Rtest)

print("εSB_old = ", εSB_old)
print("εSB_new = ", εSB_new)
print("Difference = ", np.round(εSB_old - εSB_new, 8))