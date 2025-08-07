import numpy as np
from multiprocessing import Pool
from Repres_utils import bm_to_graph,find_all_paths,distmat,append_dict,build_i_idx,\
                         ordered_charges

from Representations import build_bond_repr,build_angle_repr,build_dihedral_repr

from functools import partial

def multi_process_repr(arr,coordinate_type,num_processes = 35):
    chunks=np.array_split(arr,num_processes)
    pool = Pool(processes=num_processes)
    results = pool.map(partial(add_repr2mols, coordinate_type=coordinate_type  ) , chunks )
    return  [item for list_ in results for item in list_]


def add_repr2mols(calcs,coordinate_type):
    mols=[]
    for calc in calcs:
        x,h_ic = calc[:-1], calc[-1]
        mol=[]
        charges,xyzcoords,BOM,idxs,q,B,g_ic=x
        molg=bm_to_graph(BOM)
        BBt=B@B.T
        i_idxs=build_i_idx(idxs)
        for b in range(len(q)):  # looping over the coordinates
            idx=idxs[b]
            if len(idx)==2 and coordinate_type=="bonds": #bond
                i,j=idx
                if not ordered_charges([charges[x] for x in molg[i]],[charges[x] for x in molg[j]] ):i,j=j,i
                if len (molg[j])<len(molg[i]): i,j=j,i # Hybridization I>J
                if charges[i]<charges[j]: i,j=j,i  # assert charge_i>charge_j
                label=(charges[i],charges[j])
                repres=build_bond_repr(charges,xyzcoords,BOM,idx,i_idxs,q,b,molg)
                repres.append(h_ic[b,b])
                is_ring= len(find_all_paths(molg,i,j))>1  # rings
                mol.append([label,is_ring,repres])
            elif len(idx)==3 and coordinate_type=="angles":
                i,j,k=idx
                if not ordered_charges([charges[x] for x in molg[i]],[charges[x] for x in molg[k]] ):i,k=k,i
                if len (molg[k])<len(molg[i]): i,k=k,i  # ensure Hybridization I> K
                if BOM[i,j]<BOM[j,k]+0.3: i,k=k,i   # ensure BO[i,j] > BO[j,k]
                if charges[i]<charges[k]: i,k=k,i   # ensure Z_i >= Z_k
                aas=(charges[i],charges[j],charges[k])
                repres=build_angle_repr(charges,xyzcoords,BOM,[i,j,k],i_idxs,q,b,molg)
                repres.append(h_ic[b,b])
                ring=len(find_all_paths(molg,i,j))>1 or len(find_all_paths(molg,j,k))>1
                mol.append([ring,aas,repres])
            elif len(idx)==4 and coordinate_type=="dihedrals": 
                i,j,k,l=idx
                if BOM[i,j]<BOM[k,l]: i,j,k,l=l,k,j,i    # BO[i,j]>BO[j,k]
                if charges[j]<charges[k] or (charges[j]==charges[k] and charges[i]<charges[l]): i,j,k,l=l,k,j,i 
                aas=(charges[i],charges[j],charges[k],charges[l])
                repres=build_dihedral_repr(charges,xyzcoords,BOM,idx,i_idxs,q,molg,b)
                repres.append(h_ic[b,b])
                mol.append([ aas,repres])
        mols.append(mol)
    return mols
