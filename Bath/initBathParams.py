import numpy as np
import matplotlib.pyplot as plt

fs2au = 41.341374575751
cminv2au = 4.55633*1e-6
eV2au = 0.036749405469679
K2au = 0.00000316678

# ωc = (24.8 / 1000) * eV2au               # cutoff frequency
# λ  = (30 / 1000) * eV2au                 # Reorganisation energy

ωc = 18 * cminv2au                  # cutoff frequency
λ  = 50 * cminv2au                  # Reorganisation energy
N  = 20                             # Number of modes needed
sampleBreadth = 5
ω = np.linspace(0.00000001,sampleBreadth*ωc, 30000)
dω = ω[1]-ω[0]

def J(ω,ωc,λ):  
    f1 = 2*λ*ωc*ω
    f2 = (ωc**2) + (ω**2)
    return f1/f2

# Fω = (1/np.pi) * np.sum(J(ω[:-1],ωc,λ)/ω[:-1]) * dω
# print(Fω/cminv2au)
# exit()

Fω = np.zeros(len(ω))
for i in range(len(ω)):
    Fω[i] = (4/np.pi) * np.sum(J(ω[:i],ωc,λ)/ω[:i]) * dω

λs = Fω[-1]
ωj = np.zeros(N)
for i in range(N):
    costfunc = np.abs(Fω-(((i-0.5)/N)*λs))
    ωj[i] = ω[np.where(costfunc == np.min(costfunc))[0]]
cj = ωj * ((λs/(2*N))**0.5)

np.savetxt(f"ωj_ωc{int(ωc/cminv2au)}_λ{int(λ/cminv2au)}_{sampleBreadth}ωc_N{N}.txt",ωj)
np.savetxt(f"cj_ωc{int(ωc/cminv2au)}_λ{int(λ/cminv2au)}_{sampleBreadth}ωc_N{N}.txt",cj)

# np.savetxt(f"ωj_MotionalNarrowing_fast_N{N}.txt",ωj)
# np.savetxt(f"cj_MotionalNarrowing_fast_N{N}.txt",cj)

# plt.plot(ωj / cminv2au, J(ωj,ωc,λ) / cminv2au)
# plt.vlines(ωj / cminv2au, np.zeros(N), 2.5*np.ones(N), color='r', lw=1)
# plt.xlim(-10,2500)
# plt.savefig("chirag_bath.png")