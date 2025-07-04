#from pyscf import gto,scf
#from pyscf import hessian
import numpy as np
ang2bohr=1.8897261246
bohr2ang=.5291772109
from berny import Berny, geomlib, coords,Geometry,Math
import berny


def BO(mf,i,j):
    P=mf.make_rdm1()
    S=mf.get_ovlp()
    PS=P@S
    return np.einsum("ij,ji",PS[i_a:i_b,j_a:j_b],PS[j_a:j_b,i_a:i_b])

def BO(P,S,aoslice,i,j):
    PS=P@S
    i_a,i_b,j_a,j_b=*aoslice[i,-2:],*aoslice[j,-2:]
    return np.einsum("ij,ji",PS[i_a:i_b,j_a:j_b],PS[j_a:j_b,i_a:i_b])

def build_BOM(P,S,aoslice):
    PS=P@S
    bom=np.zeros((len(aoslice),len(aoslice)))
    for i in range(len(aoslice)):
        for j in range(len(aoslice)):
            if i==j: continue
            i_a,i_b,j_a,j_b=*aoslice[i,-2:],*aoslice[j,-2:]
            bom[i,j]=np.einsum("ij,ji",PS[i_a:i_b,j_a:j_b],PS[j_a:j_b,i_a:i_b])
    return bom