# Partial Linearised Density Matrix (PLDM)
A collection of serial, vectorized and parallel version of PLDM dynamics. The reduced density matrix is propagated by propagating the MMST variables. The individual MMST variables form a forward ($|\psi_F\rangle$) and a backward wavefunction ($\langle\psi_B|$), which can be used to decompose the $ρ_{sys}$ dynamics ($\mathbb{O} (N^2)$) into propagation of two wavefucntions ($\mathbb{O} (N)$) which exponentially reduces the computational cost of large systems. We apply this on some model polaritonic HTC hamiltonians for large number of molecules and show the speedup of the current implementation compared to previous PLDM implementations. The code is structured as 

### Bath 
- `initBathParams.py` generate the bath parameters $c_j$ and $\omega_j$ from a spectral density
- `initBathMPI.py` generates the the initial nuclei position and momenta for a trajectory

### System
- `model.py` contains the system functions
    - system Hamiltonian (Hψ)
    - Dipole (μψ)
    - system-bath interation hamiltonian (H_SB)
- `parameters.py` contains the parameters of the system and also simulation parameters

- Dynamics
- Run
- Data
- PostProcess
- Tests
