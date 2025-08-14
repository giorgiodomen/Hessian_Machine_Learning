import matplotlib.pyplot as plt

from Representations_nondiagonal import  angle_angle_consec_repr, angle_angle_adj_repr,angle_angle_vert_repr,\
bond_angle_incl_repr,bond_angle_adj_repr,bond_bond_repr
from Repres_utils import bonds_to_graph,find_all_paths,append_dict,build_i_idx, ordered_charges,distmat
from numpy import array as np_array ,diag,zeros,array_split,arange

from Representations_diagonal import  build_bond_repr, build_angle_repr ,build_dihedral_repr
from joblib import load as jl_load
from time import time

from Constants import ang2bohr,bohr2ang
from Charge2Symbol import charge


from multiprocessing import Pool
from functools import partial

str_idx={2:"bonds",3:"angles",4:"dihedrals"}

def idxstr(idx):
    rs=''
    for i in idx:
        rs+=str(i)+'_'
    return rs[:-1]

from copy import deepcopy
def dict_of_lists_merge(x, y):
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        z[key] = x[key]+ y[key]
    for key in x.keys() - overlapping_keys:
        z[key] = deepcopy(x[key])
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z


# for parellel read
def predict_Diagonal(chunk,path_models):
    preds=[]
    for rep_ic in chunk:
        rep,ic =rep_ic[:-1],rep_ic[-1]
        try:
            mod=jl_load(path_models+f"{str_idx[len(ic)]}/{idxstr(ic)}.joblib")
            mod.n_jobs=1
            preds.append([[x[1] for x in rep], mod.predict([x[0] for x in rep])])
        except:
            preds.append([[x[1] for x in rep], np_array([0.02 for x in rep])])
    return preds
def predict_nonDiagonal(chunk,path_models):
    preds=[]
    for rep_ic in chunk:
        rep,ic =rep_ic[:-1],rep_ic[-1]
        try:
            mod=jl_load(path_models+f"{ic[0]}/{idxstr(ic[1])}.joblib")
            mod.n_jobs=1
            preds.append([[x[1] for x in rep], mod.predict([x[0] for x in rep])])

        except:
            print("Found no model for ND coordinate: ",ic[0],ic[1],path_models+f"{ic[0]}/{idxstr(ic[1])}.joblib")
                    
    return preds
        
def multi_process_pred(a_dict,diagonal,num_processes = 4,path_models='./Saved_Models/'):
    K,V=list(a_dict.keys()),list(a_dict.values())
    for i in range(len(V)):
        V[i].append(K[i])
    V=np_array(V,dtype=object)
    chunks=array_split(V,num_processes)
    pool = Pool(processes=num_processes)
    if diagonal:
        results = pool.map(partial(predict_Diagonal,path_models=path_models) , chunks )
    else:
        results = pool.map(partial(predict_nonDiagonal,path_models=path_models) , chunks )
    return  [item for list_ in results for item in list_]



def repr_Diagonal(sset_coords,charges,xyzcoords,idxs,q,molg,i_idxs ):
    DM=distmat(xyzcoords)
    mol_d={}
    for b in sset_coords:
        idx=idxs[b]
        if len(idx)==2:
            i,j=idx
            if not ordered_charges([charges[x] for x in molg[i]],[charges[x] for x in molg[j]] ):i,j=j,i
            if len (molg[j])<len(molg[i]): i,j=j,i # Hybridization I>J
            if charges[i]<charges[j]: i,j=j,i  # assert charge_i>charge_j
            aas=(charges[i],charges[j])
            append_dict(mol_d,aas,[build_bond_repr(charges,xyzcoords,idx,molg,DM),b])
        elif len(idx)==3: 
            i,j,k=idx
            if not ordered_charges([charges[x] for x in molg[i]],[charges[x] for x in molg[k]] ):i,k=k,i
            if len (molg[k])<len(molg[i]): i,k=k,i  # ensure Hybridization I > K
            if charges[i]<charges[k]: i,k=k,i   #ensure Z_i>=Z_k        aas=(charges[i],charges[j],charges[k])
            aas=(charges[i],charges[j],charges[k])
            angrepr=build_angle_repr(charges,xyzcoords,[i,j,k],i_idxs,q,b,molg)
            append_dict(mol_d,aas,[angrepr,b])

        elif len(idx)==4: 
            i,j,k,l=idx
            if charges[j]<charges[k] or (charges[j]==charges[k] and charges[i]<charges[l]): i,j,k,l=l,k,j,i 
            aas=(charges[i],charges[j],charges[k],charges[l])
            repres=build_dihedral_repr(charges,xyzcoords,idx,q,molg,b)
            append_dict(mol_d, aas,[repres,b])
    return mol_d



def repr_non_Diagonal(sset_coords,charges,xyzcoords,idxs,q,molg,i_idxs,h_ic=None ):
    mol_nd={}
    for b in sset_coords:  # b index coordinate
        idx=idxs[b]
        if len(idx)==2:
            i,j=idx
        #BB_adj  V
            molgi=molg[i].copy()
            molgj=molg[j].copy()
            for k in molgi:  #k-i-j
                a2=i
                if k<=j:continue # avoid double counting
                if charges[k]>=charges[j]: a1,a3=k,j
                else: a1,a3=j,k
                cycl_class=(len(find_all_paths(molg,a1,a2)), len(find_all_paths(molg,a2,a3)))
                rv=np_array([*cycl_class,*bond_bond_repr(charges,xyzcoords,(a1,a2,a3),molg)])
                append_dict(mol_nd, ("BB_adj",tuple(charges[x] for x in (a1,a2,a3))),[rv, (b,i_idxs[(i,k)])])
            for k in molgj:  #i-j-k
                a2=j
                if k<=i:continue # avoid double counting 
                if charges[i]>=charges[k]:a1,a3=i,k
                else: a1,a3=k,i
                cycl_class=(len(find_all_paths(molg,a1,a2)), len(find_all_paths(molg,a2,a3)))
                rv=np_array([*cycl_class,*bond_bond_repr(charges,xyzcoords,(a1,a2,a3),molg)])
                append_dict(mol_nd, ("BB_adj",tuple(charges[x] for x in (a1,a2,a3))),[rv, (b,i_idxs[(j,k)])])
        elif len(idx)==3:
            # Bond Angle adjacent V
            i,j,k=idx
            if charges[i]<charges[k]: i,k=k,i  #charge i> charge k
            molgj=molg[j].copy()
            if k in molgj: molgj.remove(k)
            if i in molgj: molgj.remove(i)
            for adj_at in molgj:
                cycl_class= (len(find_all_paths(molg,i,k)),len(find_all_paths(molg,i,adj_at)),\
                             len(find_all_paths(molg,j,adj_at)))
                rv=np_array([*cycl_class,*bond_angle_adj_repr(charges,xyzcoords,(k,j,i,adj_at),molg)])
                append_dict(mol_nd, ("BA_adj",tuple(charges[x] for x in (i,j,k,adj_at))),[rv,(b,i_idxs[(j,adj_at)])])

        #  Bond angle included   i=j-k
            cycl_class= (len(find_all_paths(molg,j,k)),len(find_all_paths(molg,i,j)),len(find_all_paths(molg,i,k)))
            rv=np_array([*cycl_class,*bond_angle_incl_repr(charges,xyzcoords,(i,j,k),molg)])
            append_dict(mol_nd, ("BA_inc",tuple(charges[x] for x in (i,j,k))),[rv,(b,i_idxs[(i,j)])])
  # Also the other border k=j-i   switch 'i' and 'k'
            cycl_class= (len(find_all_paths(molg,j,i)),len(find_all_paths(molg,k,j)), len(find_all_paths(molg,k,i)))
            rv=np_array([*cycl_class,*bond_angle_incl_repr(charges,xyzcoords,(k,j,i),molg)])
            append_dict(mol_nd, ("BA_inc",tuple(charges[x] for x in (k,j,i))),[rv,(b,i_idxs[(k,j)])])  

       #Angle Angle adjacent
            i,j,k=idx
            molgj=molg[j].copy()
            if k in molgj: molgj.remove(k)
            if i in molgj: molgj.remove(i)
            if len (molgj)>0:
                for l in molgj:
                    a1,a2=i,j # the shared side of the adjacent angle is (a1-a2=i-j)
                    if l>k: #avoid double counting
                        if charges[l]>charges[k]: a3,a4=l,k
                        else: a3,a4=k,l
                        if (a1,a2,a3) in i_idxs and (a1,a2,a4) in i_idxs:
                            cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                            len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),
                                        len(find_all_paths(molg,a3,a4)))
                            rv=np_array([*cycl_class,*angle_angle_adj_repr(charges,xyzcoords,(a1,a2,a3,a4),molg)])
                            append_dict(mol_nd, ("AA_adj",tuple(charges[x] for x in (a1,a2,a3,a4))),[rv,(i_idxs[(a1,a2,a3)],i_idxs[(a1,a2,a4)])])
                    a1,a2=k,j #the shared side of the adjacent angle is (a1-a2 = k-j)
                    if l>i:  #avoid double counting
                        if charges[l]>charges[i]: a3,a4=l,i
                        else: a3,a4=i,l
                        if (a1,a2,a3) in i_idxs and (a1,a2,a4) in i_idxs:
                            cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                            len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),len(find_all_paths(molg,a3,a4)))
                            rv=np_array([*cycl_class,*angle_angle_adj_repr(charges,xyzcoords,(a1,a2,a3,a4),molg)])
                            append_dict(mol_nd, ("AA_adj",tuple(charges[x] for x in (a1,a2,a3,a4))),[rv,(i_idxs[(a1,a2,a3)],i_idxs[(a1,a2,a4)])])

        # Angle Angle consecutive
            i,j,k=idx
            molgi=molg[i].copy()
            if j in molgi:molgi.remove(j)
            if k in molgi: molgi.remove(k)
            molgk=molg[k].copy()
            if j in molgk:molgk.remove(j)
            if i in molgk: molgk.remove(i)
            if len (molgi)>0:     # Extend from the MolgI side 
                for l in molgi:
                    if l>k:  #avoid double-counting
                        if charges[l]>charges[k]: a1,a2,a3,a4=l,i,j,k
                        else: a1,a2,a3,a4=k,j,i,l
                        if (a1,a2,a3)  in i_idxs and (a2,a3,a4) in i_idxs:
                            cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                            len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),
                                        len(find_all_paths(molg,a3,a4)))
                            rv=np_array([*cycl_class,*angle_angle_consec_repr(charges,xyzcoords, (a1,a2,a3,a4),molg)])
                            append_dict(mol_nd, ("AA_consec",tuple(charges[x] for x in (a1,a2,a3,a4))),[rv,(b,i_idxs[(l,i,j)] )])
            if len (molgk)>0:  # Extend from the MolgK side
                for l in molgk:
                    if l>i:  #avoid double-counting
                        if charges[l]>charges[i]: a1,a2,a3,a4=l,k,j,i
                        else: a1,a2,a3,a4=i,j,k,l
                        if (a1,a2,a3) in i_idxs and (a2,a3,a4) in i_idxs: 
                            cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                            len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),
                                        len(find_all_paths(molg,a3,a4)))
                            rv=np_array([*cycl_class,*angle_angle_consec_repr(charges,xyzcoords,(a1,a2,a3,a4),molg)])
                            append_dict(mol_nd, ("AA_consec",tuple(charges[x] for x in (a1,a2,a3,a4))),[rv,(b,i_idxs[(l,k,j)])])
            # Angle angle Vertex
            i,v,j=idx
            molgv= molg[v].copy()
            if i in molgv:molgv.remove(i)
            if j in molgv:molgv.remove(j)
            if len (molgv)!=2: continue
            k,l=molgv[0],molgv[1]
            if max(k,l)>max(i,j):continue #avoid double repres.
            if max(charges[k],charges[l])>max(charges[i],charges[j]) or max(charges[k],charges[l])==max(charges[i],charges[j]) and min(charges[k],charges[l])>min(charges[i],charges[j]) :
                if charges[k]>charges[l]:
                    a1,a2=k,l
                else: a1,a2=l,k
                if charges[i]>charges[j]:
                    a3,a4=i,j
                else: a4,a3=i,j
            else:
                if charges[k]>=charges[l]:
                    a3,a4=k,l
                else: a3,a4=l,k
                if charges[i]>=charges[j]:
                    a1,a2=i,j
                else: a2,a1=i,j 
                if (a1,v,a2) not in i_idxs or (a3,v,a4) not in i_idxs: continue             
                rv=np_array([*angle_angle_vert_repr(charges,xyzcoords,(a1,a2,a3,a4,v),molg)])
                append_dict(mol_nd, ("AAV",tuple(charges[x] for x in (a1,a2,a3,a4,v))),[rv,(b,i_idxs[(l,v,k)])]) 
        elif len(idx)==4:
            i,j,k,l=idx
            if (i,j) in i_idxs:   
                rv=[*build_dihedral_repr(charges,xyzcoords,(i,j,k,l),q,molg,b)]
                append_dict(mol_nd, ("DB_arm",tuple(charges[x] for x in (i,j,k,l))),[np_array(rv),(b,i_idxs[(i,j)])])
            if (k,l) in i_idxs:
                rv=[*build_dihedral_repr(charges,xyzcoords,(l,k,j,i),q,molg,b)]
                append_dict(mol_nd, ("DB_arm",tuple(charges[x] for x in (l,k,j,i))),[np_array(rv),(b,i_idxs[(l,k)])])
            if charges[l]>charges[i] or (charges[i]==charges[l] and charges[k]>charges[j]):
                i,j,k,l=l,k,j,i
            if (j,k) in i_idxs:
                rv=[*build_dihedral_repr(charges,xyzcoords,(i,j,k,l),q,molg,b)]
                append_dict(mol_nd, ("DB_core",tuple(charges[x] for x in (i,j,k,l))),[np_array(rv),(b,i_idxs[(j,k)])])

    return mol_nd

def multi_process_repr(charges,xyzcoords,idxs,q,molg,i_idxs,diagonal,num_processes = 4):
    chunks=array_split(arange(len(q)),num_processes)
    pool = Pool(processes=num_processes)
    if diagonal:
        results = pool.map(partial(repr_Diagonal,\
                   charges=charges,xyzcoords=xyzcoords,idxs=idxs,q=q,molg=molg,i_idxs=i_idxs) , chunks )
    else:
        results = pool.map(partial(repr_non_Diagonal,\
                   charges=charges,xyzcoords=xyzcoords,idxs=idxs,q=q,molg=molg,i_idxs=i_idxs) , chunks )    
    val_dict=results[0]
    for d in results[1:]:
        val_dict=dict_of_lists_merge(val_dict,d)
    return  val_dict







def make_Hess(charges,xyzcoords,idxs,q,bonds,verbose=False,\
              num_processes = 1,path_models='../Saved_Models/'):
    if path_models[-1]!='/':path_models+='/'
    mol_d={}
    molg=bonds_to_graph(bonds)
    i_idxs=build_i_idx(idxs)
    t0=time()
    N_ICs=len(q)
    mol_d=multi_process_repr(charges,xyzcoords,idxs,q,molg,i_idxs,True,\
                             num_processes = num_processes)
    t1=time()
    if verbose: print(t1-t0, " diagrepres")
    HD=zeros(N_ICs)
    
    diag_predictions=multi_process_pred(mol_d,True,num_processes,path_models=path_models)
    for dp in diag_predictions:
        for i in range(len(dp[0])):
            HD[dp[0][i]]=dp[1][i]
            
    init_h= diag(HD)
    t2=time()
    if verbose: print(t2-t1, " diag pred")
    
    mol_nd=multi_process_repr(charges,xyzcoords,idxs,q,molg,i_idxs,False,\
                              num_processes = num_processes)
    t3=time()
    if verbose: print(t3-t2, " NON diag repres")
    
    nd_pred=multi_process_pred(mol_nd,False,num_processes)
    for dp in nd_pred:
        for i in range(len(dp[0])):
            init_h[dp[0][i]]=dp[1][i]*2  # *2 because after we evaluate H=(H+H.T)/2
            
    t4=time()
    if verbose: print(t4-t3, "non diag predictions")    
    
    init_h=(init_h+init_h.T)/2

    return init_h
