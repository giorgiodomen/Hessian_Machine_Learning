import argparse
import numpy as np
import os,sys
sys.path.append('./Libs')
from geometric.internal import PrimitiveInternalCoordinates
from geometric.internal import Distance,Angle,Dihedral,OutOfPlane
from geometric.molecule import Molecule as gtMolecule


def hess_2_ic(calc):
        name,xyz,elements,atcharges,g,H=calc
        GT_mol=gtMolecule()
        GT_mol.Data['elem'],GT_mol.Data['xyzs'],GT_mol.Data['comms']= elements,  [xyz],['']
        GT_mol.build_topology(force_bonds=True)
        bonds=np.array(GT_mol.bonds,dtype=(int,int))
        PICs=PrimitiveInternalCoordinates(GT_mol,True)
        h_ic=PICs.calcHess(xyz, g.flatten(), H)
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
            print('non canonical IC in ', name,ignore)
            g_ic,h_ic=g_ic[:-len(ignore)],h_ic[:-len(ignore),:-len(ignore)]
        return [name,atcharges,xyz,bonds,idxs,q,g_ic,h_ic]

def file_2_ic(fp):
    """ fp: (str) full path to file.
    Parse a .npy file containing a list of molecular data
     in the format   [name,xyz,elements,atcharges,g,H]
     name: (str) the name of the molecule 
     xyz: (array 3*N_atm) nuclear coordinates (in Angstrom)
     elements:(str N_atm) a list of atomic symbols
     atcharges: (int N_atm) a list of atomic charges (a.u.)
     g (float array 3*N_atm ) the geometrical gradient (a.u.) 
     H (float array (3*N_atm,3*N_atm) ) the geometrical Hessian (a.u.)
    """
    mols_cc=np.load(fp,allow_pickle=True)
    mols_ic=[]
    for calc in mols_cc[:]:
        mols_ic.append(hess_2_ic(calc))
    fn=fp.split('/')[-1][:-4]
    np.save(f'./IC_data/IC_{fn}.npy', np.array(mols_ic,dtype=object))    
    print('saved files under ',f'./IC_data/IC_{fn}.npy')


if __name__ == "__main__":
    try:
        os.mkdir('IC_data')
    except FileExistsError:
        print('folder \'IC_data\' already exists !')
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name")
    args = parser.parse_args()
    if os.path.exists(args.file_name):
        print(args.file_name)
        file_2_ic(args.file_name)