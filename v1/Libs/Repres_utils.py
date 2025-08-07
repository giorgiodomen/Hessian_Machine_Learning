import numpy as np
from numpy import dot,cross
from numpy.linalg import norm
import matplotlib.pyplot as plt

def ordered_charges(ci,cj):
    while len(ci)>len(cj):
        cj.append(0)
    while len(cj)>len(ci):
        ci.append(0)
    ci.sort(reverse=True),cj.sort(reverse=True)
    for n,i in enumerate(ci):
        if i>cj[n]:
            return True
        elif cj[n]>i:
            return False
    return True

def dihedral_cos(coords,idx):
    """ 2 cos (alpha)+1
    """
    i,j,k,l=idx
    v1 = (coords[i] - coords[j]) 
    v2 = (coords[l] - coords[k]) 
    w = (coords[k] - coords[j])
    ew = w / norm(w)
    a1 = v1 - np.dot(v1, ew) * ew
    a2 = v2 - np.dot(v2, ew) * ew
    det=np.linalg.det(np.array([v2, v1, w]))
    if det<1e-3:
        if v1.dot(v2)>0: return 2   # Check code 
        else: return 0
    dot_product = np.dot(a1, a2) / (norm(a1) * norm(a2))  #cos (dih)
    if dot_product < -1:
        dot_product = -1
    elif dot_product > 1:
        dot_product = 1  
    if np.isnan(dot_product): print (v1,v2,a1,a2,w,norm(w),np.linalg.det(np.array([v2, v1, w])))
    return 1+dot_product
def angle_cos(coords,idx):
    """ 2 cos (alpha)+1
    """
    i,j,k=idx
    v1 = (coords[i] - coords[j]) 
    v2 = (coords[k] - coords[j]) 
    dot_product = np.dot(v1, v2) / (norm(v1) * norm(v2))
    if dot_product < -1:
        dot_product = -1
    elif dot_product > 1:
        dot_product = 1  
    if np.isnan(dot_product): print (v1,v2,a1,a2,w,norm(w),np.linalg.det(np.array([v2, v1, w])))
    return 1+dot_product

def get_dihedral(coords,idx):
    i,j,k,l=idx
    v1 = (coords[i] - coords[j]) 
    v2 = (coords[l] - coords[k])
    w = (coords[k] - coords[j])
    det=np.linalg.det(np.array([v2, v1, w]))
    if det<1e-3:
        return 0
    ew = w / norm(w)
    a1 = v1 - dot(v1, ew) * ew
    a2 = v2 - dot(v2, ew) * ew
    sgn = np.sign(np.linalg.det(np.array([v2, v1, w])))
    sgn = sgn or 1

    dot_product = dot(a1, a2) / (norm(a1) * norm(a2))
    if dot_product < -1:
        dot_product = -1
    elif dot_product > 1:
        dot_product = 1
    phi = np.arccos(dot_product) * sgn    
    return phi

def bm_to_graph(BOM):
    cm=BOM>0.5
    cm_graph={}
    for i in range(len(cm)): 
        conn_list=[]
        for j in range(len(cm[i])):
            if cm[i,j]:
                conn_list.append(j)
        cm_graph[i]=conn_list
    return cm_graph
def mol_integrity(idxs,molg):
    return  np.all([integrity(idx,molg) for idx in idxs])
def integrity(idx,molg):
    if len(idx)==2:
        if  idx[0] in molg[idx[1]]:
            return True
        else: 
            return False
    if not (idx[0] in molg[idx[1]] and idx[2] in molg[idx[1]]):
        return False
    if len(idx)==4:
           if not (idx[1] in molg[idx[2]] and idx[3] in molg[idx[2]]):
                return False 
    return (True)

def find_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not start in graph:
            return None
        for node in graph[start]:
            if node not in path:
                newpath = find_path(graph, node, end, path)
                if newpath: return newpath
        return None

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not start in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths    

def distmat(xyzcoords):
    xyzcoords=np.array([xyzcoords])
    extcm=np.vstack([xyzcoords]*xyzcoords.shape[0])
    distm=extcm-extcm.swapaxes(0,1)#.shape
    return np.einsum("ijk,ijk->ij",distm,distm)**.5+100*np.eye(xyzcoords.shape[1])

def append_dict(dictionary,key,value):
        try: dictionary[key].append(value)
        except: dictionary[key]=[value]
def build_i_idx(idxs):
    i_idxs={}
    for n_e,idx_e in enumerate(idxs):
        i_idxs[idx_e]=n_e
        i_idxs[idx_e[::-1]]=n_e
    return i_idxs

def BM_CM(charges,xyzcoords,i,j,k):
    distm=distmat(xyzcoords)
    distm[:,i]=100
    distm[:,j]=100
    distm[:,k]=100
    iV,jV,kV=distm[i],distm[j],distm[k]
    dbmat=np.array([1/iV**2*BOM[:,i]*charges,1/jV**2*BOM[:,j]*charges,1/kV**2*BOM[:,k]*charges])
    dbmat.resize(3,20)
    dbmatS=np.sort(dbmat)[:,-1::-1]
    dbmat=np.sort(dbmat,axis=0)
    return dbmat

