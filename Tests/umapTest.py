import numpy as np
import parameters as param
from numba import jit

import testHel as model
import parameters as param


@jit(nopython=True)
def Umap_old(qF, qB, pF, pB, VMat):
    ψF = (1 / np.sqrt(2)) * (qF - 1j * pF)
    ψB = (1 / np.sqrt(2)) * (qB - 1j * pB)
    
    # updating cF
    ψF1 = VMat @ ψF
    ψF2 = 2 * (VMat @ ψF1) - ψF
    ψF3 = 2 * (VMat @ ψF2) - ψF1
    ψFt = (param.b[0] * ψF[1:param.NMol+2]) + (param.b[1] * ψF1[1:param.NMol+2]) + (param.b[2] * ψF2[1:param.NMol+2]) + (param.b[3] * ψF3[1:param.NMol+2])
    ψFt = ψFt * param.expF
    ψFΔ = ψF[:]
    ψFΔ[1:param.NMol+2] = ψFt[:]
    
    
    # updating cB
    ψB1 = VMat @ ψB
    ψB2 = 2 * (VMat @ ψB1) - ψB
    ψB3 = 2 * (VMat @ ψB2) - ψB1
    ψBt = (param.b[0] * ψB[1:param.NMol+2]) + (param.b[1] * ψB1[1:param.NMol+2]) + (param.b[2] * ψB2[1:param.NMol+2]) + (param.b[3] * ψB3[1:param.NMol+2])
    ψBt = ψBt * param.expF
    ψBΔ = ψB[:]
    ψBΔ[1:param.NMol+2] = ψBt[:]
    
    qF, pF = np.real(ψFΔ) * np.sqrt(2), -np.imag(ψFΔ) * np.sqrt(2)
    qB, pB = np.real(ψBΔ) * np.sqrt(2), -np.imag(ψBΔ) * np.sqrt(2)
    
    return qF, qB, pF, pB


@jit(nopython=True)
def Umap_new(qF, qB, pF, pB, ε_sb):
    
    ψF  = (1 / np.sqrt(2)) * (qF - 1j * pF) 
    ψB  = (1 / np.sqrt(2)) * (qB - 1j * pB)
    
    # updating cF
    ψF1 = model.Hψ(ψF, ε_sb)
    ψF2 = 2 * model.Hψ(ψF1, ε_sb) - ψF
    ψF3 = 2 * model.Hψ(ψF2, ε_sb) - ψF1
    ψFt = (param.b[0] * ψF[1:param.NMol+2]) + (param.b[1] * ψF1[1:param.NMol+2]) + (param.b[2] * ψF2[1:param.NMol+2]) + (param.b[3] * ψF3[1:param.NMol+2])
    ψFt = ψFt * param.expF
    
    ψFΔ = ψF[:]
    ψFΔ[1:param.NMol+2] = ψFt[:]
    
    
    # updating cB
    ψB1 = model.Hψ(ψB, ε_sb)
    ψB2 = 2 * model.Hψ(ψB1, ε_sb) - ψB
    ψB3 = 2 * model.Hψ(ψB2, ε_sb) - ψB1
    ψBt = (param.b[0] * ψB[1:param.NMol+2]) + (param.b[1] * ψB1[1:param.NMol+2]) + (param.b[2] * ψB2[1:param.NMol+2]) + (param.b[3] * ψB3[1:param.NMol+2])
    ψBt = ψBt * param.expF
    
    ψBΔ = ψB[:]
    ψBΔ[1:param.NMol+2] = ψBt[:]
    
    
    qF[:], pF[:] = (np.real(ψFΔ) * np.sqrt(2))[:], (-np.imag(ψFΔ) * np.sqrt(2))[:]
    qB[:], pB[:] = (np.real(ψBΔ) * np.sqrt(2))[:], (-np.imag(ψBΔ) * np.sqrt(2))[:]
    

R =  np.random.random(param.NR) * 1000
ε_SB = model.H_SB(R)

Vmat = model.Hel_Chebyshev(R)

qF_old = np.random.rand(param.NMol+2)
qB_old = np.random.rand(param.NMol+2)
pF_old = np.random.rand(param.NMol+2)
pB_old = np.random.rand(param.NMol+2)

qF_new = qF_old.copy()
qB_new = qB_old.copy()
pF_new = pF_old.copy()
pB_new = pB_old.copy()


qF_old_test, qB_old_test, pF_old_test, pB_old_test = Umap_old(qF_old, qB_old, pF_old, pB_old, Vmat)
Umap_new(qF_new, qB_new, pF_new, pB_new, ε_SB)

print("Real part")
print("qF diff : ", np.round(qF_new.real - qF_old_test.real, 16))
print("qB diff : ", np.round(qB_new.real - qB_old_test.real, 16)) 
print("pF diff : ", np.round(pF_new.real - pF_old_test.real, 16))
print("pB diff : ", np.round(pB_new.real - pB_old_test.real, 16))
print("\n")

print("Imaginary part")
print("qF diff : ", np.round(qF_new.imag - qF_old_test.imag, 16))
print("qB diff : ", np.round(qB_new.imag - qB_old_test.imag, 16)) 
print("pF diff : ", np.round(pF_new.imag - pF_old_test.imag, 16))
print("pB diff : ", np.round(pB_new.imag - pB_old_test.imag, 16))
print("\n")