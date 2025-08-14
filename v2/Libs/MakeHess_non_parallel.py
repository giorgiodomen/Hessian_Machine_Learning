import matplotlib.pyplot as plt


from Representations_nondiagonal import  angle_angle_consec_repr, angle_angle_adj_repr,angle_angle_vert_repr,\
bond_angle_incl_repr,bond_angle_adj_repr,bond_bond_repr,dihedral_bond_arms,dihedral_bond_core,build_DB_repr
from Repres_utils import bm_to_graph,find_all_paths,append_dict,build_i_idx,ordered_charges
from numpy import array as np_array ,hstack,eye

from Representations_diagonal import  build_bond_repr, build_angle_repr ,build_dihedral_repr
from joblib import load as jl_load
from time import time
from Constants ang2bohr,bohr2ang

def idxstr(idx):
    rs=''
    for i in idx:
        rs+=str(i)
    return rs

def make_Hess(charges,xyzcoords,BOM,idxs,q):
    mol=[]
    molg=bm_to_graph(BOM)
    i_idxs=build_i_idx(idxs)
    Total_bonds=0
    t0=time()
    for b in range(len(q)):  # need index coordinates
        idx=idxs[b]
        if len(idx)==2:
            i,j=idx
            if not ordered_charges([charges[x] for x in molg[i]],[charges[x] for x in molg[j]] ):i,j=j,i
            if len (molg[j])<len(molg[i]): i,j=j,i # Hybridization I>J
            if charges[i]<charges[j]: i,j=j,i  # assert charge_i>charge_j
            aas=(charges[i],charges[j])
            is_ring=len(find_all_paths(molg,i,j))>1  # rings
            mol.append([is_ring,aas,build_bond_repr(charges,xyzcoords,BOM,idx,i_idxs,q,b,molg),(b,b)])
            Total_bonds+=1
        elif len(idx)==3: 
            i,j,k=idx
            if not ordered_charges([charges[x] for x in molg[i]],[charges[x] for x in molg[k]] ):i,k=k,i
            if len (molg[k])<len(molg[i]): i,k=k,i  # ensure Hybridization I > K
            if BOM[i,j]<BOM[j,k]+0.3: i,k=k,i   # ensure BO[i,j]>BO[j,k]
            if charges[i]<charges[k]: i,k=k,i   #ensure Z_i>=Z_k        aas=(charges[i],charges[j],charges[k])
            aas=(charges[i],charges[j],charges[k])
            angrepr=build_angle_repr(charges,xyzcoords,BOM,[i,j,k],i_idxs,q,b,molg)
            is_ring=len(find_all_paths(molg,i,j))>1 or len(find_all_paths(molg,j,k))>1
            mol.append([is_ring,aas,angrepr,(b,b)])

        elif len(idx)==4: 
            i,j,k,l=idx
            if BOM[i,j]<BOM[k,l]: i,j,k,l=l,k,j,i    # BO[i,j]>BO[j,k]
            if charges[j]<charges[k] or (charges[j]==charges[k] and charges[i]<charges[l]): i,j,k,l=l,k,j,i 
            aas=(charges[i],charges[j],charges[k],charges[l])
            repres=build_dihedral_repr(charges,xyzcoords,BOM,idx,i_idxs,q,molg,b)
            is_ring=not (repres[4]==1 and repres[5]==1)
            mol.append([is_ring, aas,repres,(b,b)])

    t1=time()
    print(t1-t0, " diagrepres")
    str_idx={2:"Bonds",3:"Angles",4:"Dihedrals"}
    to_ring={True:"ring",False:"lin"}
    
    gdm=jl_load(f"/home/giorgio/Documents/HPML/HPML/Final_models/Saved_Models/Dihedral_general.joblib")
    keyset=set()
    for ic in mol:
        keyset.add((ic[0],ic[1]))
    models={}
    for ic in keyset:
        try:
            models[ic]=jl_load(f"../Diagonal/Saved_Models/{str_idx[len(ic[1])]}/{to_ring[ic[0]]}_{idxstr(ic[1])}.joblib")
        except:
            print("No model found for ", ic )
            pass
    HD=[]
    for de in mol:
        try:
            if len(de[1]) in [2,3]:
                HD.append(models[(de[0],de[1])].predict([de[2]])[0])
                #print(len(de[2]))
            elif len(de[1])==4:
                HD.append(models[(de[0],de[1])].predict([de[2]])[0])
                #print(len(de[2]))
        except:
            HD.append(gdm.predict(hstack([np_array(de[1]),de[2]]).reshape(1,-1))[0])
    init_h= eye(len(np_array(HD).reshape(-1)))*(np_array(HD).reshape(-1))

    t2=time()
    print(t2-t1, " diag pred")
    mol_nd=[]
    molg=bm_to_graph(BOM)
    i_idxs=build_i_idx(idxs)

    for b in range(len(q)):  # need index coordinates
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
                rv=np_array([*cycl_class,*bond_bond_repr(charges,xyzcoords,BOM,(a1,a2,a3),i_idxs,q,molg)])
                mol_nd.append( ["BB_adj",tuple(charges[x] for x in (a1,a2,a3)),rv, (b,i_idxs[(i,k)])])
            for k in molgj:  #i-j-k
                a2=j
                if k<=i:continue # avoid double counting 
                if charges[i]>=charges[k]:a1,a3=i,k
                else: a1,a3=k,i
                cycl_class=(len(find_all_paths(molg,a1,a2)), len(find_all_paths(molg,a2,a3)))
                rv=np_array([*cycl_class,*bond_bond_repr(charges,xyzcoords,BOM,(a1,a2,a3),i_idxs,q,molg)])
                mol_nd.append(["BB_adj",tuple(charges[x] for x in (a1,a2,a3)),rv, (b,i_idxs[(j,k)])])

        elif len(idx)==3:
            i,j,k=idx
            if charges[i]<charges[k]: i,k=k,i  #charge i> charge k
        # Bond Angle adjacent V
            molgj=molg[j].copy()
            if k in molgj: molgj.remove(k)
            if i in molgj: molgj.remove(i)
            for adj_at in molgj:
                cycl_class= (len(find_all_paths(molg,i,k)),len(find_all_paths(molg,i,adj_at)),\
                             len(find_all_paths(molg,j,adj_at)))
                rv=np_array([*cycl_class,*bond_angle_adj_repr(charges,xyzcoords,BOM,(k,j,i,adj_at),i_idxs,molg,q)])
                mol_nd.append(["BA_adj",tuple(charges[x] for x in (i,j,k,adj_at)),rv,(b,i_idxs[(j,adj_at)])])

        #  Bond angle included   i=j-k
            cycl_class= (len(find_all_paths(molg,j,k)),len(find_all_paths(molg,i,j)),\
                                 len(find_all_paths(molg,i,k)))
            rv=np_array([*cycl_class,*bond_angle_incl_repr(charges,xyzcoords,BOM,(i,j,k),i_idxs,molg,q)])
            mol_nd.append(["BA_inc",tuple(charges[x] for x in (i,j,k)),rv,(b,i_idxs[(i,j)])])
            # Also the other border k=j-i
            cycl_class= (len(find_all_paths(molg,j,i)),len(find_all_paths(molg,k,j)),\
                                 len(find_all_paths(molg,k,i)))
            rv=np_array([*cycl_class,*bond_angle_incl_repr(charges,xyzcoords,BOM,(k,j,i),i_idxs,molg,q)])
            mol_nd.append(["BA_inc",tuple(charges[x] for x in (k,j,i)),rv,(b,i_idxs[(k,j)])])  

       #Angle Angle adjacent
            if len (molgj)>0:
                for l in molgj:
                 # K eq to a3 
                    a1,a2=i,j #atomo al centro (a2=j) e quello condiviso (a1=i)
                    if l>k:
                        if charges[l]>charges[k]: a3,a4=l,k
                        else: a3,a4=k,l
                        cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                        len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),
                                     len(find_all_paths(molg,a3,a4)))
                        rv=np_array([*cycl_class,*angle_angle_adj_repr(charges,xyzcoords,BOM,(a1,a2,a3,a4),i_idxs,molg,q)])
                        mol_nd.append(["AA_adj",tuple(charges[x] for x in (a1,a2,a3,a4)),rv,(i_idxs[(a1,a2,a3)],i_idxs[(a1,a2,a4)])])

                    # I eq to a3 
                    a1,a2=k,j #atomo al centro (a2=j) e quello condiviso (a1=k)
                    if l>i:
                        if charges[l]>charges[i]: a3,a4=l,i
                        else: a3,a4=i,l
                        cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                        len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),
                                     len(find_all_paths(molg,a3,a4)))
                        rv=np_array([*cycl_class,*angle_angle_adj_repr(charges,xyzcoords,BOM,(a1,a2,a3,a4),i_idxs,molg,q)])
                        mol_nd.append(["AA_adj",tuple(charges[x] for x in (a1,a2,a3,a4)),rv,(i_idxs[(a1,a2,a3)],i_idxs[(a1,a2,a4)])])

        # Angle Angle consecutive
                molgi=molg[i].copy()
                if j in molgi:molgi.remove(j)
                if k in molgi: molgi.remove(k)
                molgk=molg[k].copy()
                if j in molgk:molgk.remove(j)
                if i in molgk: molgk.remove(i)
            # Extend from the MolgI side
                if len (molgi)>0:
                    for l in molgi:
                # Portare ad    a1-a2-a3 // a2-a3-a4
                        if l>k:  #avoid double-counting
                            if charges[l]>charges[k]: a1,a2,a3,a4=l,i,j,k
                            else: a1,a2,a3,a4=k,j,i,l
                            cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                            len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),
                                         len(find_all_paths(molg,a3,a4)))
                            rv=np_array([*angle_angle_consec_repr(charges,xyzcoords,BOM,(a1,a2,a3,a4),i_idxs,molg,q)])
                            mol_nd.append(["AA_consec",tuple(charges[x] for x in (a1,a2,a3,a4)),rv,(b,i_idxs[(l,i,j)] )])
            # Extend from the MolgK side
                if len (molgk)>0:
                    for l in molgk:
                # Portare ad    a1-a2-a3-a4
                        if l>i:  #avoid double-counting
                            if charges[l]>charges[i]: a1,a2,a3,a4=l,k,j,i
                            else: a1,a2,a3,a4=i,j,k,l
                            cycl_class= (len(find_all_paths(molg,a2,a1)),len(find_all_paths(molg,a2,a3)),\
                            len(find_all_paths(molg,a1,a3)),len(find_all_paths(molg,a1,a4)),
                                         len(find_all_paths(molg,a3,a4)))
                            rv=np_array([*angle_angle_consec_repr(charges,xyzcoords,BOM,(a1,a2,a3,a4),i_idxs,molg,q)])
                            mol_nd.append(["AA_consec",tuple(charges[x] for x in (a1,a2,a3,a4)),rv,(b,i_idxs[(l,k,j)])])
            # Angle angle Vertex
            i,v,j=idx
            molgv= molg[v].copy()
            if i in molgv:molgv.remove(i)
            if j in molgv:molgv.remove(j)
            if len (molgv)!=2: continue
            k,l=molgv[0],molgv[1]
            if max(k,l)>max(i,j):continue #avoid double repres.
            if max(charges[k],charges[l])>max(charges[i],charges[j]) or \
    max(charges[k],charges[l])==max(charges[i],charges[j]) and min(charges[k],charges[l])>min(charges[i],charges[j]) :
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
                rv=np_array([*angle_angle_vert_repr(charges,xyzcoords,BOM,(a1,a2,a3,a4,v),i_idxs,molg,q)])
                mol_nd.append(["AAV",tuple(charges[x] for x in (a1,a2,a3,a4,v)),rv,(b,i_idxs[(l,v,k)])]) 
        elif len(idx)==4:
            i,j,k,l=idx   
            rv=[*dihedral_bond_arms(charges,xyzcoords,BOM,(i,j,k,l),q,b)]
            rv=rv+[*build_DB_repr(charges,xyzcoords,BOM,(i,j,k,l),i_idxs,q,molg,b)]
            mol_nd.append(["DB_arm",tuple(charges[x] for x in (i,j,k,l)),np_array(rv),(b,i_idxs[(i,j)])])
            rv=[*dihedral_bond_arms(charges,xyzcoords,BOM,(l,k,j,i),q,b)]
            rv=rv+[*build_DB_repr(charges,xyzcoords,BOM,(l,k,j,i),i_idxs,q,molg,b)]
            mol_nd.append(["DB_arm",tuple(charges[x] for x in (l,k,j,i)),np_array(rv),(b,i_idxs[(l,k)])])
            rv=[*dihedral_bond_core(charges,xyzcoords,BOM,(i,j,k,l),q,b)]
            rv=rv+[*build_DB_repr(charges,xyzcoords,BOM,(i,j,k,l),i_idxs,q,molg,b)]
            if charges[l]>charges[i] or (charges[i]==charges[l] and charges[k]>charges[j]):
                i,j,k,l=l,k,j,i
            mol_nd.append(["DB_core",tuple(charges[x] for x in (i,j,k,l)),np_array(rv),(b,i_idxs[(j,k)])])
    t3=time()
    print(t3-t2, " NON diag repres")
    
    keyset_nd=set()
    for ic in mol_nd:
        keyset_nd.add((ic[0],ic[1]))
    #start parallization
    models_nd={}
    for ic in keyset_nd:
        try:
            models_nd[ic]=jl_load(f"../Nondiagonal/Saved_Models/{ic[0]}/lin_{idxstr(ic[1])}.joblib")
        except:
            try:
                models_nd[ic]=jl_load(f"../Nondiagonal/Saved_Models/{ic[0]}/{idxstr(ic[1])}.joblib")
            except:
                print(ic[0],ic[1])
            models_nd[ic].njobs=1
    t4=time()
    print(t4-t3, "non diag load")    
    for de in mol_nd:
        try:
            init_h[de[-1]] =(models_nd[(de[0],de[1])].predict([de[2]]))*2  # *2 because after we do H=(H+H.T)/2
        except:
            init_h[de[-1]]  =0
    init_h=(init_h+init_h.T)/2
    t5=time()
    print(t5-t4, "non diag predictions")  
    return init_h
