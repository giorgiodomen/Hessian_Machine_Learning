import numpy as np

import sys 
from Repres_utils import distmat,dihedral_cos,angle_cos,find_all_paths,ordered_charges


def build_bond_repr(charges,xyzcoords,BOM,idx,i_idxs,q,b,molg):
    DM=distmat(xyzcoords)
    i,j=idx
    #env I
    molgi=molg[i].copy()
    if j in molgi: molgi.remove(j)
    env_i=np.zeros(3*4)
    envi_list=[]
    for atom in molgi:
        envi_list.append([charges[atom],BOM[atom,i],q[i_idxs[(atom,i)]],\
                          angle_cos(xyzcoords,(j,i,atom))] )
    
    envi_list.sort(reverse=True)
    envi_list=np.asarray(envi_list).flatten()
    env_i[:min(12,len(envi_list))] =envi_list[:min(12,len(envi_list))]
    #env J
    molgj=molg[j].copy()
    if i in molgj: molgj.remove(i)
    env_j=np.zeros(3*4)
    envj_list=[]
    for atom in molgj:
        envj_list.append([charges[atom],BOM[atom,j],q[i_idxs[(atom,j)]],\
                          angle_cos(xyzcoords,(i,j,atom))] )
    envj_list.sort(reverse=True)
    envj_list=np.asarray(envj_list).flatten()
    env_j[:min(12,len(envj_list))] =envj_list[:min(12,len(envj_list))]
    return [q[b],BOM[i,j],*env_i,*env_j]



def build_angle_repr(charges,xyzcoords,BOM,idx,i_idxs,q,b,molg):
    DM=distmat(xyzcoords)
    i,j,k=idx
    bloblo= [DM[i,j],DM[j,k],BOM[i,j],BOM[j,k],BOM[i,k]]  #angle width 
    molgj=molg[j].copy()
    if k in molgj: molgj.remove(k),
    if i in molgj: molgj.remove(i)
    z_j_len=10
    adj_blolo=[]
    for atom in molgj:
        adj_blolo.append([charges[atom],BOM[atom,j],DM[atom,j],\
                                  angle_cos(xyzcoords,(k,j,atom)),dihedral_cos(xyzcoords,(i,j,k,atom))])
    z_j=np.zeros(z_j_len)
    adj_blolo.sort(reverse=True)
    adj_blolo=np.asarray(adj_blolo).flatten()
    z_j[:min(z_j_len,len(adj_blolo))]=adj_blolo[:min(z_j_len,len(adj_blolo))]
    repr_len=15
    molgj_ord=[]
    for atom in molgj:
        molgj_ord.append([charges[atom],BOM[atom,j],atom])
        molgj_ord.sort(reverse=True)
    if charges[k] !=1:  # non H 
        molgk=molg[k].copy()
        if j in molgk: molgk.remove(j)
        adj_blolo=[]
        for atom in molgk:
                adj_blolo.append([charges[atom],BOM[atom,k],DM[atom,k],\
                                  angle_cos(xyzcoords,(j,k,atom)),dihedral_cos(xyzcoords, (i,j,k,atom)) ] )
    else:  # If k is Hydrogen
        adj_blolo=[]
        if len(molgj_ord)>=1:
            a1=molgj_ord[0][2]
            molga1=molg[a1].copy()
            if j in molga1: molga1.remove(j)
            for atom in molga1:
                adj_blolo.append([charges[atom],BOM[atom,a1], DM[atom,a1],\
                          angle_cos(xyzcoords,(j,a1,atom)),dihedral_cos(xyzcoords, (i,j,a1,atom))])
    z_k=np.zeros(repr_len)
    adj_blolo.sort(reverse=True)
    adj_blolo=(np.asarray(adj_blolo)).flatten()
    z_k[:min(repr_len,len(adj_blolo))]=adj_blolo[:min(repr_len,len(adj_blolo))]  
    if charges[i] !=1: # non H
        molgi=molg[i].copy()
        if j in molgi: molgi.remove(j)
        adj_blolo=[]
        for atom in molgi:
            adj_blolo.append([charges[atom],BOM[atom,i],DM[atom,i],\
                              angle_cos(xyzcoords,(j,i,atom)),dihedral_cos(xyzcoords, (k,j,i,atom))] )
    else:    # If i is Hydrogen
        adj_blolo=[]
        n_h=( charges[k] ==1)+(charges[k] ==1)
        if len(molgj_ord)>=n_h:
            a1=molgj_ord[n_h-1][2]
            molga1=molg[a1].copy()
            molga1.remove(j) 
            for atom in molga1:
                    adj_blolo.append([charges[atom],BOM[atom,a1], DM[atom,a1],\
                                  angle_cos(xyzcoords,(j,a1,atom)),dihedral_cos(xyzcoords,(i,j,a1,atom))] )
    adj_blolo.sort(reverse=True)  
    adj_blolo=(np.asarray(adj_blolo)).flatten()
    z_i=np.zeros(repr_len)
    z_i[:min(repr_len,len(adj_blolo))]=adj_blolo[:min(repr_len,len(adj_blolo))]
    repres=[1+np.cos(q[b]),*bloblo,*z_j,*z_i,*z_k]
    return repres


def build_dihedral_repr(charges,xyzcoords,BOM,idx,i_idxs,q,molg,b):
    DM=distmat(xyzcoords)
    i,j,k,l=idx
    bloabloa=[DM[i,j],BOM[i,j],angle_cos(xyzcoords,(i,j,k)), DM[j,k], BOM[j,k],angle_cos(xyzcoords,(j,k,l)),\
              DM[k,l], BOM[k,l] ] 
    ringdisp=[len(find_all_paths(molg, i,j)),len(find_all_paths(molg, j,k)),\
              len(find_all_paths(molg, k,l)),len(find_all_paths(molg, i,k)),\
              len(find_all_paths(molg, j,l)),len(find_all_paths(molg, i,l))]
    
    molgi=molg[i].copy()
    molgi.remove(j)
    molgj=molg[j].copy()
    molgj.remove(k),molgj.remove(i)
    molgk=molg[k].copy()
    molgk.remove(j),molgk.remove(l)
    molgl=molg[l].copy()
    molgl.remove(k)
    
    za=np.zeros(50)
    lim=[0,15,25,35]
    for n_g,molg_n in enumerate([molgi,molgj,molgk,molgl ]):
        adj_l=[]
        if len(molg_n)>0:
            for atom in molg_n:
                adj_l.append([charges[atom],BOM[atom,idx[n_g]],DM[atom,idx[n_g]], \
        angle_cos(xyzcoords,(atom,idx[n_g],idx[n_g+1] if n_g<2 else idx[n_g-1])),\
dihedral_cos(xyzcoords,(atom,idx[n_g],idx[n_g+1] if n_g<2 else idx[n_g-1],idx[n_g+2] if n_g<2 else idx[n_g-2]))])
            adj_l.sort(reverse=True)
            adj_l=[x for ar in adj_l for x in ar]
            za[lim[n_g]:lim[n_g]+len(adj_l)]=np.asarray(adj_l)
    repres=[1+np.cos(q[b]),*ringdisp,*bloabloa,*za]
    return repres
