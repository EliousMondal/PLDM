import numpy as np
import torch
import model
import parameters as param


def Force(data, F):
    """
        Description → calculating the force on the nuclear degrees of freedom
                      from the bath and the system-bath interaction
            F = - ∑ⱼωⱼ²Rⱼ - (1/4)∑ᵢⱼcⱼ(qFᵢ² + pFᵢ² + qBᵢ² + pBᵢ²) 
    """
    data.ρii[:, :] = 0.0
    F[:, :]        = 0.0    
    
    data.ρii[:, :]  = data.qF[1: data.nmols+1, :] ** 2 + data.pF[1: data.nmols+1, :] ** 2 + data.qB[1: data.nmols+1, :] ** 2 + data.pB[1: data.nmols+1, :] ** 2
    F[: ,:]         = - torch.einsum('i, ij -> ij', data.ωj_nr_sq, data.R)
    F[:, :]        -= 0.25 * torch.einsum('ij, k -> ikj', data.ρii, data.cj).reshape(data.NR, data.NCols)
    # i: np.int32
    # for i in range(data.NCols):
    #     F[:, i] = F[:, i] - 0.25 * torch.kron(data.ρii[:, i], data.cj)


def Umap(data):
    """
        Description → updating the mapping variables for the forward and backward 
                      wavefunctions via chebyshev polynomial expansion
                    |F(t+δ)⟩ = eⁱᴴᵟ|F(t)⟩ and  ⟨B(t+δ)| = ⟨B(t)|eⁱᴴᵟ
            with            |F⟩ = (1/√2)(|qF⟩ - i|pF⟩)
            and             ⟨B| = (1/√2)(⟨qB| - i⟨pB|)
    """
    data.ψF[:, :]  = (1 / np.sqrt(2)) * (data.qF[:, :] - 1j * data.pF[:, :]) 
    data.ψB[:, :]  = (1 / np.sqrt(2)) * (data.qB[:, :] - 1j * data.pB[:, :])
    
    # updating cF
    data.ψF1[:, :] = model.Hψ(data.ψF, data)
    data.ψF2[:, :] = 2 * model.Hψ(data.ψF1[:, :], data) - data.ψF
    data.ψF3[:, :] = 2 * model.Hψ(data.ψF2[:, :], data) - data.ψF1[:, :]
    data.ψFt[:, :] = (param.b[0] * data.ψF[1:data.nstates, :]) + (param.b[1] * data.ψF1[1:data.nstates, :]) + (param.b[2] * data.ψF2[1:data.nstates, :]) + (param.b[3] * data.ψF3[1:data.nstates, :])
    data.ψFt[:, :] = data.ψFt[:, :] * param.expF
    
    data.ψFΔ[:, :] = data.ψF[:, :]
    data.ψFΔ[1:data.nstates, :] = data.ψFt[:, :]
    
    # updating cB
    data.ψB1[:, :] = model.Hψ(data.ψB, data)
    data.ψB2[:, :] = 2 * model.Hψ(data.ψB1[:, :], data) - data.ψB
    data.ψB3[:, :] = 2 * model.Hψ(data.ψB2[:, :], data) - data.ψB1[:, :]
    data.ψBt[:, :] = (param.b[0] * data.ψB[1:data.nstates, :]) + (param.b[1] * data.ψB1[1:data.nstates, :]) + (param.b[2] * data.ψB2[1:data.nstates, :]) + (param.b[3] * data.ψB3[1:data.nstates, :])
    data.ψBt[:, :] = data.ψBt[:, :] * param.expF
    
    data.ψBΔ[:, :] = data.ψB[:, :]
    data.ψBΔ[1:data.nstates, :] = data.ψBt[:, :]
    
    # getting the updated mapping variables
    data.qF[:, :], data.pF[:, :] = (torch.real(data.ψFΔ[:, :]) * np.sqrt(2)), (-torch.imag(data.ψFΔ[:, :]) * np.sqrt(2))
    data.qB[:, :], data.pB[:, :] = (torch.real(data.ψBΔ[:, :]) * np.sqrt(2)), (-torch.imag(data.ψBΔ[:, :]) * np.sqrt(2))


def VelVer(data):
    """
        Description → updating the phonon R, P with velocity verlet integration method
    """
    data.v = data.P/param.M
    
    model.H_SB(data)
    Umap(data)

    Force(data, data.F1)
    data.R  = data.R + data.v * param.dtN + 0.5 * data.F1 * param.dtN ** 2 / param.M
    Force(data, data.F2)
    data.v  = data.v + 0.5 * (data.F1 + data.F2) * param.dtN / param.M
    data.P  = data.v * param.M
    
    model.H_SB(data)
    Umap(data) 


def runTraj(data):
    
    model.initMapping(data)          # initialise mapping variables
    
    data.qFt[:, :] *= 0.0
    data.qBt[:, :] *= 0.0
    data.pFt[:, :] *= 0.0
    data.pBt[:, :] *= 0.0
    
    iskip = 0 
    i: np.int                      
    for i in range(data.nsteps):
        if (i % param.nskip == 0):
            data.qFt[iskip, :, :] += data.qF[:, :]
            data.qBt[iskip, :, :] += data.qB[:, :]
            data.pFt[iskip, :, :] += data.pF[:, :]
            data.pBt[iskip, :, :] += data.pB[:, :]
            iskip += 1
            
        VelVer(data)