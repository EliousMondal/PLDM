import numpy as np
import numba as nb
import sys

import parameters as param 
import model

@nb.jit(nopython=True)
def ρij(qF, pF, qB, pB, ρ0):
    return np.kron(qF + 1j * pF, qB-1j*pB) * ρ0

ρsize = param.NStates * param.NStates
ρt0 = 0.25 * (1 - 1j) * (1 - 1j)
ρ = np.zeros((param.NSaveStps, 2 * param.NStates * param.NStates))

# dipole test
operators = np.array([[0, i] for i in range(1, param.NMol+1)], dtype=int)


print("Loading data ...")
TrajFolder = sys.argv[1]

for op in operators:
    iF, iB = op[0], op[1]
    
    qFt = np.loadtxt(TrajFolder + f"{1}/qFt_{1}_{iF}{iB}.txt")
    qBt = np.loadtxt(TrajFolder + f"{1}/qBt_{1}_{iF}{iB}.txt")
    pFt = np.loadtxt(TrajFolder + f"{1}/pFt_{1}_{iF}{iB}.txt")
    pBt = np.loadtxt(TrajFolder + f"{1}/pBt_{1}_{iF}{iB}.txt")

    t_axis = np.linspace(0, param.SimTime, qFt.shape[0])

    print(f"Computing density matrx for |{iF}⟩⟨{iB}|...")
    ρt = np.zeros((qFt.shape[0], 2 * ρsize))
    for i in range(t_axis.shape[0]):
        ρt_i = ρij(qFt[i, :], pFt[i, :], qBt[i, :], pBt[i, :], ρt0)
        ρt[i, :ρsize], ρt[i, ρsize:] = ρt_i.real, ρt_i.imag

    print("Saving Data ...")
    np.savetxt(TrajFolder + f"{1}/rho_t_{iF}{iB}.txt", ρt, fmt="% 24.16e")