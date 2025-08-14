import numpy as np
import sys,os,argparse

sys.path.append("./Libs")

from Charge2Symbol import charge as sym2charge

from Constants import ang2bohr

from MakeHess import make_Hess

import geometric
from geometric.internal import PrimitiveInternalCoordinates
from geometric.internal import Distance,Angle,Dihedral,OutOfPlane
from geometric.molecule import Molecule as gtMolecule
from geometric.molecule import Elements

from sklearn.ensemble import RandomForestRegressor

def parse_g_xtb(fn):
    gradient=[]
    with open(fn,'r') as f:
        lines=f.readlines()
        n_atm= (len(lines)-3)//2
        lines=lines[n_atm+2:-1]
        for l in lines:
            for f in l.split()[:]:
                gradient.append(float(f))
    gradient=np.array(gradient)
    return gradient



def predict_hess(xyz_file_path,grad_file_path,saved_models,n_jobs,output):
    GT_mol=gtMolecule(xyz_file_path)
    bonds=np.array(GT_mol.bonds,dtype=(int,int))
    PICs=PrimitiveInternalCoordinates(GT_mol,True)
    atcharges=[sym2charge(a) for a in GT_mol.elem]
    xyz=GT_mol.xyzs[0]*ang2bohr
    if grad_file_path:
        g=parse_g_xtb(grad_file_path)
    else:
        g=np.zeros_like(xyz)
    g_ic = PICs.calcGrad(xyz, g.flatten())
    idxs=[]
    q=[]
    ignore=[]
    for p in PICs.Internals:
        if type(p) is Distance:
            idxs.append((p.a,p.b))
        elif type(p) is Angle:
            idxs.append((p.a,p.b,p.c))            
        elif type(p) is Dihedral or type(p) is OutOfPlane:
            idxs.append((p.a,p.b,p.c,p.d))
        else:
            ignore.append(p)
            continue
        q.append(p.value(xyz))
    if len(ignore)!= 0:
        print(name,ignore)
        g_ic,h_ic=g_ic[:-len(ignore)],h_ic[:-len(ignore),:-len(ignore)]
    h_p=make_Hess(atcharges,xyz,idxs,q,bonds,num_processes = n_jobs,path_models=saved_models)
    hp_CC=PICs.calcHessCart(xyz, g_ic, h_p)
    xyzfilename=(xyz_file_path.split('/')[-1])[:-4]
    np.save(output+xyzfilename,hp_CC)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xyzfile", help=" xyz file")
    parser.add_argument("-g","--gradient", default=None,help='path to file or none')
    parser.add_argument("-s","--saved_models", default='./Saved_Models/',help='where the models are saved')
    parser.add_argument("-o","--output", default='./predicted_Hess/',help='where to save Hessian')
    parser.add_argument("-j","--n_jobs", default=3,help='Number Jobs')
    args = parser.parse_args()
    print(args.xyzfile,args.gradient,args.saved_models,args.n_jobs,args.output)
    try:os.mkdir(args.output)
    except FileExistsError: pass
    predict_hess(args.xyzfile,args.gradient,args.saved_models,args.n_jobs,args.output)