import numpy as np


from functools import partial
from multiprocessing import Pool


from sympy_derivs import bond_hess,angle_hess,dihedral_hess,sign_dihedral
from Dihedral_B_derivs import a_fun,a_derivs,grad_deriv,grad_deriv_1,grad_deriv_2
from Angle_B_derivs import ang_trasf_mat,ang_grad_1,ang_grad_2

from Constants import *
from berny import Geometry,Math,Berny
from Charge2Symbol import to_cm


def Build_h_ic(at_coords,atoml,g,H_cart,num_processes=1):
    at_coords=at_coords.copy()
    g=np.asarray(g).reshape(-1)
    if not len(H_cart.shape)==2:
        H_cart=H_cart.swapaxes(1,2)
        H_cart=H_cart.reshape(len(g),len(g))
    geom0=Geometry(atoml,at_coords*bohr2ang)
    bernyobj=Berny(geom0)
    s=bernyobj._state
    Bmat = s.coords.B_matrix(geom0)
    B_inv = Bmat.T.dot(symmMatInv(np.dot(Bmat, Bmat.T)))
    g_ic=np.dot(B_inv.T, g)
    g_x=Bmat.T@g_ic
    if num_processes==1:
        BpG=build_BpG_cord(atoml,at_coords,g_ic,list(s.coords),np.arange(len(s.coords)) )
    else:
        BpG=build_BpG(atoml,at_coords,s,g_ic,num_processes)
    h_ic=B_inv.T@(H_cart-BpG)@B_inv
    return h_ic,g_ic,s

def symmMatInv(A,th=5e-4):
    dim = A.shape[0]
    det = 1.0
    evals, evects = np.linalg.eigh(A)
    evects = evects.T
    for i in range(dim):
        det *= evals[i]
    diagInv = np.zeros( (dim,dim), float)
    for i in range(dim):
        if abs(evals[i]) > th:
            diagInv[i,i] = 1.0/evals[i]            
    # A^-1 = P^t D^-1 P
    tmpMat = np.dot(diagInv, evects)
    AInv = np.dot(evects.T, tmpMat)
    return AInv


def build_BpG_cord(atoml,at_coords,g_ic,IC_coords,IC_idxs) :
    BpG=np.zeros((len(atoml)*3,len(atoml)*3))
    for i in IC_idxs:  #indexes
        coord=IC_coords[i]
        idx=coord.idx
        if len(idx)==2:
            i1,i2=idx
            for j,q in enumerate([*range(3*i1,3*i1+3),*range(3*i2,3*i2+3)]):
                for k,p in enumerate([*range(3*i1,3*i1+3),*range(3*i2,3*i2+3)]):
                    BpG[q,p]+=bond_hess[j][k](*at_coords[i1],*at_coords[i2])*g_ic[i]
        if len(idx)==3:
            i1,i2,i3=idx
            v1v2=ang_trasf_mat.T.dot(np.hstack([at_coords[i1],at_coords[i2],at_coords[i3]]))
            dot_product=v1v2[:3].dot(v1v2[3:])/np.linalg.norm(v1v2[:3])/np.linalg.norm(v1v2[3:])
            if dot_product < -1:        dot_product = -1
            elif dot_product > 1:        dot_product = 1
            phi = np.arccos(dot_product)
            ang_grad_deriv=np.zeros((6,9))
            if abs(phi) > np.pi - 1e-4:
                if abs(dot_product)>.99:
                    amax,amin=np.argmax(v1v2[:3]),np.argmin(v1v2[:3])
                    v1v2[amax]-=v1v2[amin]*1e-7
                    v1v2[amin]+=v1v2[amax]*1e-7
                    for j in range(6):
                        for k in range(9):
                            ang_grad_deriv[j][k]= ang_grad_2[j][k](*v1v2)
            else:
                for j in range(6):
                    for k in range(9):
                        ang_grad_deriv[j][k]= ang_grad_1[j][k](*v1v2)
            dih_increment=np.dot(ang_trasf_mat,ang_grad_deriv)*g_ic[i]
            for j,j_idx in enumerate([*range(3*i1,3*i1+3),*range(3*i2,3*i2+3),*range(3*i3,3*i3+3)]):
                for k,k_idx in enumerate([*range(3*i1,3*i1+3),*range(3*i2,3*i2+3),*range(3*i3,3*i3+3)]):
                    BpG[j_idx,k_idx]+=dih_increment[j,k]
        if len(idx)==4:
            i1,i2,i3,i4=idx
            red_coords= a_fun(*at_coords[i1],*at_coords[i2],*at_coords[i3],*at_coords[i4])
            Ader=np.zeros((11,12))
            for k in range(11):
                for j in range(12):
                    Ader[k,j]=a_derivs[k][j](*at_coords[i1],*at_coords[i2],*at_coords[i3],*at_coords[i4])
            a1,a2=np.array(red_coords[:3]),np.array(red_coords[3:6])
            dot_product = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
            if dot_product < -1:
                dot_product = -1
            elif dot_product > 1:
                dot_product = 1
            phi = np.arccos(dot_product) 
            Gradder=np.zeros((12,11))
            if abs(phi) > np.pi - 1e-5:
                for k in range(12):
                    for j in range(11):
                        Gradder[k,j]=grad_deriv_1[k][j](*red_coords)
            elif abs(phi) < 1e-5:
                for k in range(12):
                    for j in range(11):
                        Gradder[k,j]=grad_deriv_2[k][j](*red_coords)
            else:
                for k in range(12):
                    for j in range(11):
                        Gradder[k,j]=grad_deriv[k][j](*red_coords)
            dih_increment=np.einsum("ij,ki->jk",Ader,Gradder) *g_ic[i] 
            for k,k_idx in enumerate([*range(3*i1,3*i1+3),*range(3*i2,3*i2+3),*range(3*i3,3*i3+3),*range(3*i4,3*i4+3)]):
                for j,j_idx in enumerate([*range(3*i1,3*i1+3),*range(3*i2,3*i2+3),*range(3*i3,3*i3+3),*range(3*i4,3*i4+3)]):
                    BpG[k_idx,j_idx]+=dih_increment[k,j]
    return BpG


def build_BpG(atoml,at_coords,s,g_ic,num_processes=12) :
    IC_idxs=np.arange(len(s.coords))
    chunks=np.array_split(IC_idxs,num_processes)
    pool = Pool(processes=num_processes)
    results=pool.map(partial(build_BpG_cord , atoml,at_coords,g_ic,list(s.coords)) ,chunks)
    return np.sum(results,axis=0)


def Build_h_cart(at_coords,atoml,g_ic,h_ic,num_processes=12):
    geom0=Geometry(atoml,at_coords*bohr2ang)
    bernyobj=Berny(geom0)
    s=bernyobj._state
    Bmat = s.coords.B_matrix(geom0)
    if num_processes==1:
        BpG=build_BpG_cord(atoml,at_coords,g_ic,list(s.coords),np.arange(len(s.coords)) )
    else:  
        BpG=build_BpG(atoml,at_coords,s,g_ic,num_processes)
    h_cart=Bmat.T@h_ic@Bmat+BpG
    return h_cart
