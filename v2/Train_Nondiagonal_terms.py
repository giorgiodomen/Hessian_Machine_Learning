import numpy as np
import sys,os,argparse
from joblib import dump as jl_dump

sys.path.append("./Libs")
from Repres_utils import append_dict,bonds_to_graph,build_i_idx
from MakeHess import repr_non_Diagonal

from sklearn.ensemble import RandomForestRegressor


def save_models(datasets,save_to,n_jobs,n_est,max_depth):
    data=[np.load(fn, allow_pickle=True) for fn in datasets]
    data=np.vstack(data)
    data=data[:,1:] #exclude names
    Representations=[]
    for calc in data[:10]:
        charges,xyz,bonds,idxs,q,g_ic,h_ic=calc
        Representations.append([repr_non_Diagonal(np.arange(len(q)),charges,xyz,idxs,q,\
                                bonds_to_graph(bonds),build_i_idx(idxs)),h_ic])
    Models={}
    Xvals={}
    Yvals={}

    for mol in Representations:
        repres,h_ic=mol
        for label in repres:
            for rv_idx in repres[label]:
                rv,idx=rv_idx
                append_dict(Xvals,label,rv)
                append_dict(Yvals,label,h_ic[idx])
    for key in Xvals:
        x_train,y_train=np.array(Xvals[key]),np.array(Yvals[key])
        rf = RandomForestRegressor(n_estimators=n_est,n_jobs=n_jobs,max_depth=max_depth)
        rf.fit(x_train, y_train)
        rf.n_jobs=1
        Models[key]=rf

    for i in Models:
        try:os.mkdir("{}/{}".format(save_to,i[0]))
        except FileExistsError: pass
        jl_dump(Models[i],"{}/{}/".format(save_to,i[0])+('{}_'*len(i[1])).format(*i[1])[:-1]+'.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='*',help=" .npy archives of Hessians in ICs")
    parser.add_argument("-s","--save_to", default='./Saved_Models',help='where to save the models')
    parser.add_argument("-j","--n_jobs", default=3,help='Number Jobs')
    parser.add_argument("-e","--n_est", default=20,help='Random forest number estimators')
    parser.add_argument("-d","--max_depth", default=20,help='Random forest max_depth')
    args = parser.parse_args()
    print(args.datasets,args.save_to,args.n_jobs,args.n_est,args.max_depth)
    try:os.mkdir(args.save_to)
    except FileExistsError: pass
    save_models(args.datasets,args.save_to,args.n_jobs,args.n_est,args.max_depth)