import numpy as np
from scipy import special

# Fundamental constant conversions
fs2au     = 41.341374575751                 # fs to au
cminv2au  = 4.55633*1e-6                    # cm⁻¹ to au
eV2au     = 0.036749405469679               # eV to au
K2au      = 0.00000316678                   # K to au

# Trajectory parameters
NTraj     = 1                               # Total number oftrajectories
SimTime   = 500                             # Total simulation time (fs)               
dtN       = 5.0                             # bath time step (au)
NSteps    = int(SimTime/(dtN/fs2au)) + 1    # Total bath steps              
nskip     = 1                               # save data every nskip steps
pl        = (NSteps%nskip == 0)           
NSaveStps = NSteps//nskip + pl              # Total number of steps data to be saved at

# System parameters
NMol      = 10                              # number of molecules
NStates   = NMol + 2                        # |G⁰⟩, |Eᵢ⁰⟩, |G¹⟩
ε         = 0.5 * eV2au                     # average exciton energy
Δ         = 0.0 * eV2au                     # detuning of cavity with ε
ω         = ε + Δ                           # energy of cavity mode
Ω         = (200 / 1000) * eV2au            # Rabi splitting
g         = Ω / (2 * np.sqrt(NMol))         # light-matter coupling
σε        = 0.00 * eV2au                    # max exciton energy inhomogenity
δε        = np.zeros(NMol) * eV2au          # exciton energy inhomogenity     
         
# Spectral density
cj        = np.loadtxt("/scratch/mmondal/specTest/Bath/cj_ωc18_λ50_5ωc_N20.txt")
ωj        = np.loadtxt("/scratch/mmondal/specTest/Bath/ωj_ωc18_λ50_5ωc_N20.txt") 

# Bath parameters
M         = 1                               # mass of nuclear particles (au)
NModes    = len(cj)                         # Number of bath modes per site
NR        = NMol * NModes                   # Total number of bath DOF
λ         = 50 * cminv2au                   # bath reorganisation energy

# Chebyshev parameters
ΔE        = Ω + (2 * λ) + σε 
E_min     = -(0.5 * Ω) - λ - (0.5 * σε) 
g_norm    = g * (2 / ΔE)
z         = (dtN / 2) * ΔE / 2
φ         = np.exp(1j * (ΔE/2 + E_min) * (dtN / 2))
expF      = np.exp(1j * ε * dtN / 2)

# Chebyshev coefficients
numChebTerms = 3
def evalCheb(i):
    if i == 0:
        return special.jv(0, z) * φ
    else:
        return special.jv(i, z) * 2 * (1j ** i) * φ
b = np.array([evalCheb(i) for i in range(numChebTerms+1)])