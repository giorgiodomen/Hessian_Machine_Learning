import numpy as np
import copy,sys,os,argparse
from joblib import dump as jl_dump
from joblib import load as jl_load
sys.path.append("./Libs")

from parallel_representations import multi_process_repr
from Repres_utils import append_dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

diag_coords=["bonds","angles","dihedrals"]

def save_models(datasets,coord_types,save_to,n_jobs,n_est,max_depth):
    data=[np.load(fn, allow_pickle=True) for fn in datasets]
    data=np.vstack(data)
    data=data[:,1:] #exclude names
    for coord_type in coord_types: 
        Representations={}
        Models={}
        mols=multi_process_repr(data[:],coord_type,num_processes = n_jobs)
        for mol in mols:
            for ic in mol:
                label,repres=ic
                append_dict(Representations,label,repres)
        for key in Representations:
            x_train,y_train=np.array(Representations[key])[:,:-1],np.array(Representations[key])[:,-1] 
            rf = RandomForestRegressor(n_estimators=n_est,n_jobs=n_jobs,max_depth=max_depth)
            rf.fit(x_train, y_train)
            rf.n_jobs=1
            Models[key]=rf
        try:os.mkdir(save_to)
        except FileExistsError: pass
        try:os.mkdir("{}/{}".format(save_to,coord_type))
        except FileExistsError: pass
        for i in Models:
            jl_dump(Models[i],"{}/{}/".format(save_to,coord_type)+('{}_'*len(i)).format(*i)[:-1]+'.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='*',help=" .npy archives of Hessians in ICs")
    parser.add_argument("-c","--coord_types", nargs='*',default=diag_coords,\
            help="a list of IC types for which train ML models ")
    parser.add_argument("-s","--save_to", default='./Saved_Models',help='where to save the models')
    parser.add_argument("-j","--n_jobs", default=3,help='Number Jobs')
    parser.add_argument("-e","--n_est", default=40,help='Random forest number estimators')
    parser.add_argument("-d","--max_depth", default=20,help='Random forest max_depth')
    args = parser.parse_args()
    print(args.datasets,args.coord_types,args.save_to,args.n_jobs,args.n_est,args.max_depth)
    save_models(args.datasets,args.coord_types,args.save_to,args.n_jobs,args.n_est,args.max_depth)