import numpy as np

import sys
sys.path.append("../Libs")
from Repres_utils import find_all_paths,distmat,angle_cos,dihedral_cos




def angle_angle_adj_repr(charges,xyzcoords,idx,molg):
    i,j,k,l=idx
    DM=distmat(xyzcoords)
    
    molgi=molg[i].copy()
    if j in molgi:molgi.remove(j)
    molgj=molg[j].copy()
    if k in molgj:molgj.remove(k),
    if i in molgj:molgj.remove(i),
    if l in molgj:molgj.remove(l)
    molgk=molg[k].copy()
    if j in molgk:molgk.remove(j)
    molgl=molg[l].copy()
    if j in molgl:molgl.remove(j)
    
    za=np.zeros(30)
    if len(molgi)>3 or len(molgj)>2 or len(molgk)>3:
        print("error!!")
    lims=[0,9,12,21]
    if len(molgj):
        for n_g,molg_n in enumerate([molgi,molgj,molgk,molgl ]):
            adj_ar=[]
            for atom in molg_n:
                if idx[n_g]!=j:
                    adj_ar.append([charges[atom],DM[atom,idx[n_g]],\
                                              angle_cos(xyzcoords,(atom,idx[n_g],j))] )
                else:
                    adj_ar.append([charges[atom],DM[atom,idx[n_g]], \
                                          angle_cos(xyzcoords,(atom,idx[n_g],i))] )
            adj_ar.sort()
            adj_ar=[x for ar in adj_ar for x in ar]
            za[lims[n_g]:lims[n_g]+len(adj_ar)]=np.asarray(adj_ar)
    
    return  DM[i,j],DM[j,k],angle_cos(xyzcoords,(i,j,k)),\
         DM[i,l],DM[j,l],angle_cos(xyzcoords,(i,j,l)),\
        DM[k,l],DM[k,i],angle_cos(xyzcoords,(i,l,k)),*za 



def angle_angle_consec_repr(charges,xyzcoords,idx,molg):
    i,j,k,l=idx
    DM=distmat(xyzcoords)
    
    molgi=molg[i].copy()
    if j in molgi:molgi.remove(j)
    molgj=molg[j].copy()
    if k in molgj:molgj.remove(k)
    if i in molgj:molgj.remove(i)
    molgk=molg[k].copy()
    if j in molgk:molgk.remove(j)
    molgl=molg[l].copy()
    if k in molgl:molgl.remove(k)
    
    za=np.zeros(30)
    lims=[0,9,12,21]
    if len(molgj):
        for n_g,molg_n in enumerate([molgi,molgj,molgk,molgl ]):
            adj_ar=[]
            for atom in molg_n:
                if idx[n_g]!=j:
                    adj_ar.append([charges[atom],DM[atom,idx[n_g]],\
                                              angle_cos(xyzcoords,(atom,idx[n_g],j))] )
                else:
                    adj_ar.append([charges[atom],DM[atom,idx[n_g]], \
                                          angle_cos(xyzcoords,(atom,idx[n_g],l))] )
            adj_ar.sort()
            adj_ar=[x for ar in adj_ar for x in ar]
            za[lims[n_g]:lims[n_g]+len(adj_ar)]=np.asarray(adj_ar)
    
    return  dihedral_cos(xyzcoords,idx),DM[i,j],DM[j,k],angle_cos(xyzcoords,(i,j,k)),\
         DM[k,l],angle_cos(xyzcoords,(j,k,l)),DM[i,l],*za 


def angle_angle_vert_repr(charges,xyzcoords,idx,molg):
    i,j,k,l,v=idx
    DM=distmat(xyzcoords)
    
    molgi=molg[i].copy()
    if v in molgi :molgi.remove(v)
    molgj=molg[j].copy()
    if v in molgj :molgj.remove(v)
    molgk=molg[k].copy()
    if v in molgk :molgk.remove(v)
    molgl=molg[l].copy()
    if v in molgl :molgl.remove(v)
    
    za=np.zeros(48) # 4*3*4
    if len(molgi)>3 or len(molgj)>3 or len(molgk)>3:
        print("error!!")
    lims=[0,12,24,36]
    if len(molgj):
        for n_g,molg_n in enumerate([molgi,molgj,molgk,molgl ]):
            adj_ar=[]
            for atom in molg_n:
                adj_ar.append([charges[atom],DM[atom,idx[n_g]],\
            angle_cos(xyzcoords,(atom,idx[n_g],v)),dihedral_cos(xyzcoords,(atom,idx[n_g],v,idx[2*(n_g<2)]))])

            adj_ar.sort()
            adj_ar=[x for ar in adj_ar for x in ar]
            za[lims[n_g]:lims[n_g]+len(adj_ar)]=np.asarray(adj_ar)
    
    return  DM[i,v],DM[j,v],angle_cos(xyzcoords,(i,v,j)),\
         DM[v,l],DM[v,k],angle_cos(xyzcoords,(k,v,l)),*za 

def bond_angle_adj_repr(charges,xyzcoords,idx,molg):
    i,j,k,aa=idx
    DM=distmat(xyzcoords)
    molgi=molg[i].copy()
    if j in molgi :molgi.remove(j)
    molgj=molg[j].copy()
    if k in molgj :molgj.remove(k)
    if i in molgj :molgj.remove(i)
    if aa in molgj :molgj.remove(aa)
    molgk=molg[k].copy()
    if j in molgk :molgk.remove(j)
    molgaa=molg[aa].copy()
    if j in molgaa :molgaa.remove(j)
    za=np.zeros(30)

    lims=[0,9,12,21,30]
    if len(molgj):
        for n_g,molg_n in enumerate([molgi,molgj,molgk,molgaa ]):
            adj_l=[]
            for atom in molg_n:
                if idx[n_g]!=j:
                    adj_l.append([charges[atom],DM[atom,idx[n_g]],\
                                              angle_cos(xyzcoords,(atom,idx[n_g],j))] )
                else:
                    adj_l.append([charges[atom],DM[atom,idx[n_g]], \
                                          angle_cos(xyzcoords,(atom,idx[n_g],aa))] )  #make angle with aa
            adj_l.sort()
            adj_l=[x for l in adj_l for x in l]
            za[lims[n_g]:lims[n_g]+len(adj_l)]=np.asarray(adj_l)
    
    return  [DM[i,j],DM[j,k],DM[aa,j], angle_cos(xyzcoords,(i,j,k)),\
             angle_cos(xyzcoords,(i,j,aa)),angle_cos(xyzcoords,(k,j,aa)),*za]  


def bond_angle_incl_repr(charges,xyzcoords,idx,molg):
    i,j,k=idx
    DM=distmat(xyzcoords)
    molgi=molg[i].copy()
    if j in molgi: molgi.remove(j)
    molgj=molg[j].copy()
    if k in molgj:molgj.remove(k)
    if i in molgj:molgj.remove(i)
    molgk=molg[k].copy()
    if j in molgk: molgk.remove(j)
    za=np.zeros(24)
    lims=[0,9,15,24]
    if len(molgj):
        for n_g,molg_n in enumerate([molgi,molgj,molgk ]):
            adj_l=[]
            for atom in molg_n:
                if idx[n_g]!=j:
                    adj_l.append([charges[atom],DM[atom,idx[n_g]],\
                                              angle_cos(xyzcoords,(atom,idx[n_g],j))] )
                else:
                    adj_l.append([charges[atom],DM[atom,idx[n_g]], \
                                          angle_cos(xyzcoords,(atom,idx[n_g],i))] )
            adj_l.sort()
            adj_l=[x for l in adj_l for x in l]
            za[lims[n_g]:lims[n_g]+len(adj_l)]=np.asarray(adj_l)
    
    return  DM[i,j],DM[j,k],angle_cos(xyzcoords,(i,j,k)),*za 


def bond_bond_repr(charges,xyzcoords,idx,molg):
    DM=distmat(xyzcoords)
    i,j,k=idx
    bloblo=[DM[i,j],DM[j,k],DM[i,k]]    #angle width 
    molgj=molg[j].copy()
    if k in molgj: molgj.remove(k)
    if i in molgj: molgj.remove(i)
    z_j_len=8
    adj_blolo=[]
    for atom in molgj:
        adj_blolo.append([charges[atom],1/DM[atom,j],angle_cos(xyzcoords,(i,j,atom)),\
                          angle_cos(xyzcoords,(k,j,atom))] )
    z_j=np.zeros(z_j_len)
    adj_blolo.sort(reverse=True)
    adj_blolo=np.asarray(adj_blolo).flatten()
    z_j[:min(z_j_len,len(adj_blolo))]=adj_blolo[:min(z_j_len,len(adj_blolo))]
    repr_len=12
    molgj_ord=[]
    for atom in molgj:
        molgj_ord.append([charges[atom],atom])
        molgj_ord.sort(reverse=True)
    if charges[k] !=1:
        molgk=molg[k].copy()
        if j in molgk: molgk.remove(j)
        adj_blolo=[]
        for atom in molgk:
                adj_blolo.append([charges[atom],1/DM[atom,k],angle_cos(xyzcoords,(j,k,atom)),\
                                  dihedral_cos(xyzcoords,(i,j,k,atom))] )
    else:  
        adj_blolo=[]
        if len(molgj_ord)>=1:
            a1=molgj_ord[0][1]
            molga1=molg[a1].copy()
            if j in molga1: molga1.remove(j)
            for atom in molga1:
                adj_blolo.append([charges[atom],(len(molga1)-1)*100,1/DM[atom,j],1/DM[atom,a1],\
                          angle_cos(xyzcoords,(j,a1,atom)),dihedral_cos(xyzcoords,(i,j,a1,atom))] )
            
    z_k=np.zeros(repr_len)
    adj_blolo.sort(reverse=True)
    adj_blolo=np.asarray(adj_blolo).flatten()
    z_k[:min(repr_len,len(adj_blolo))]=adj_blolo[:min(repr_len,len(adj_blolo))]
    if charges[i] !=1:
        molgi=molg[i].copy()
        if j in molgi: molgi.remove(j)
        adj_blolo=[]
        for atom in molgi:
            adj_blolo.append([charges[atom],1/DM[atom,i],angle_cos(xyzcoords,(j,i,atom)),\
                              dihedral_cos(xyzcoords,(k,j,i,atom))] )
        adj_blolo.sort(reverse=True)
    else:  
        adj_blolo=[]
        if len(molgj_ord)>=2:
            a1=molgj_ord[1][1]
            molga1=molg[a1].copy()
            molga1.remove(j) 
            for atom in molga1:
                    adj_blolo.append([charges[atom],(len(molga1)-1)*100,1/DM[atom,a1],\
                                      angle_cos(xyzcoords,(j,a1,atom)), dihedral_cos(xyzcoords,(i,j,a1,atom))] )
            
    adj_blolo.sort(reverse=True)    
    adj_blolo=np.asarray(adj_blolo).flatten()
    z_i=np.zeros(repr_len)
    z_i[:min(repr_len,len(adj_blolo))]=adj_blolo[:min(repr_len,len(adj_blolo))]
    repres=[angle_cos(xyzcoords,(i,j,k)),len(molg[j])-1,*bloblo,*z_k,*z_i,*z_j]
    return repres


def build_DB_repr(charges,xyzcoords,idx,q,molg,b):
    DM=distmat(xyzcoords)
    i,j,k,l=idx
    bloabloa=[DM[i,j],angle_cos(xyzcoords,(i,j,k)), DM[j,k], angle_cos(xyzcoords,(j,k,l)),\
              DM[k,l] ] 
    ringdisp=[len(find_all_paths(molg, i,j)),len(find_all_paths(molg, j,k)),\
              len(find_all_paths(molg, k,l)),len(find_all_paths(molg, i,k)),\
              len(find_all_paths(molg, j,l)),len(find_all_paths(molg, i,l))]
    
    molgi=molg[i].copy()
    if j in molgi :molgi.remove(j)
    molgj=molg[j].copy()
    if k in molgj :molgj.remove(k)
    if i in molgj :molgj.remove(i)
    molgk=molg[k].copy()
    if j  in molgk :molgk.remove(j)
    if  l in molgk :molgk.remove(l)
    molgl=molg[l].copy()
    if k  in molgl :molgl.remove(k)

    za=np.zeros(32)
    lims=[0,6,10,14,20]

    for n_g,molg_n in enumerate([molgi,molgj,molgk,molgl ]):
        adj_l=[]
        if len(molg_n)>0:
            for atom in molg_n:
                adj_l.append([charges[atom],DM[atom,j]])
        adj_l.sort()
        adj_l=[x for ar in adj_l for x in ar]
        za[lims[n_g]:lims[n_g]+len(adj_l)]=np.asarray(adj_l)

    repres=[-np.cos(q[b]),*ringdisp,*bloabloa,*za,len(molg[j])-1]
    return repres

def dihedral_bond_core(charges,xyzcoords,BOM,idx,q,b):
    i,j,k,l=idx
    distm=distmat(xyzcoords)
    iV,jV,kV,lV=distm[i],distm[j],distm[k],distm[l]
    dbmat=np.array([1/iV**2*BOM[:,i]*charges,1/jV**2*BOM[:,j]*charges,1/kV**2*BOM[:,k]*charges,\
                    1/lV**2*BOM[:,l]*charges])
    dbmat.resize(4,21)
    dbmatS=np.sort(dbmat)[:,-1::-1]
    dbmat=np.sort(dbmat,axis=0)
    wblolo=np.asarray([q[b],*dbmatS.flatten()] )
    return wblolo

def dihedral_bond_arms(charges,xyzcoords,BOM,idx,q,b):
    i,j,k,l=idx
    distm=distmat(xyzcoords)
    iV,jV,kV,lV=distm[i],distm[j],distm[k],distm[l]
    dbmat=np.array([1/iV**2*BOM[:,i]*charges,1/jV**2*BOM[:,j]*charges,1/kV**2*BOM[:,k]*charges,\
                    1/lV**2*BOM[:,l]*charges])
    dbmat.resize(4,21)
    dbmatS=np.sort(dbmat)[:,-1::-1]
    dbmat=np.sort(dbmat,axis=0)
    wblolo=np.asarray([q[b],*dbmatS.flatten()] )
    return wblolo
