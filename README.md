# Partial Linearised Density Matrix (PLDM)
A collection of serial, vectorized and parallel version of PLDM dynamics. The reduced density matrix is propagated by propagating the MMST variables. The individual MMST variables form a forward ($|\psi_F\rangle$) and a backward wavefunction ($\langle\psi_B|$), which can be used to decompose the $ρ_{sys}$ dynamics ($\mathcal{O}$($N^2$)) into propagation of two wavefucntions ($\mathcal{O}$($N$)) which exponentially reduces the computational cost of large systems. We apply this on some model polaritonic HTC hamiltonians for large number of molecules and show the speedup of the current implementation compared to previous PLDM implementations. The code is structured as 

### Bath 
- `initBathParams.py` generate the bath parameters $c_j$ and $\omega_j$ from a spectral density
- `initBathMPI.py` generates the the initial nuclei position and momenta for a trajectory

### System
- `model.py` contains the system functions
    - system Hamiltonian (Hψ)
    - Dipole (μψ)
    - system-bath interation hamiltonian (H_SB)
- `parameters.py` contains the parameters of the system and also simulation parameters

### Dynamics
- `pldm.py` contains the dynamics functions
    - Force  → calculates the force on the bath DOF's from the system dynamics
    $$ F(R_i)  = c_j^i \sum_i \rho_{ii}$$
    - Umap   → propagates the mapping variables according to chebyshev expansion
    $$  e^{-i\frac{\hat{H}_{\mathrm{HTC}}\Delta}{\hbar}}\ket{\Psi}  = b_0(z)|\Psi^{(0)}\rangle + \sum_{n=1}^{\infty} \phi_n(z) b_n(z) |\Psi^{(n)}\rangle $$
    - velver → propagates the nuclei 
- `trajClass.py` defines a trajectory class


- Run
- Data
- PostProcess
- Tests
