import numpy as np
import matplotlib.pyplot as plt
from time import time
import igraph as ig
import modbp
import forceatlas2 as fa2
import seaborn as sbn
import sklearn.metrics as skm
import pandas as pd
import multiprocessing as mp
import itertools
import gzip
import pickle
from contextlib import contextmanager


@contextmanager
def terminating(obj):
    '''
    Context manager to handle appropriate shutdown of processes
    :param obj: obj to open
    :return:
    '''
    try:
        yield obj
    finally:
        obj.terminate()


def _run_modbp_multilayer(n_c_eta_ep_q_nlayers_gamma_omega_ntrials):
    #we return a single number averaged over the number of trials
    n,c,eta,ep,q,nlayers,resgamma,omega,ntrials=n_c_eta_ep_q_nlayers_gamma_omega_ntrials

    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(q) * pin + (np.ones((q, q)) - np.identity(q)) * pout

    allstats=pd.DataFrame(columns=['ret_mod','ami','accuracy'],dtype=float)

    for i in range(ntrials):
        #run it for the number of trials
        ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
        mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.interedges, ml_sbm.layer_vec,
                                   comm_vec=ml_sbm.get_all_layers_block())
        mlbp = modbp.ModularityBP(mlgraph=mgraph)
        bstar=mlbp.get_bstar(q=q)
        mlbp.run_modbp(beta=bstar,q=q,resgamma=resgamma,omega=omega,niter=500)
        allstats.loc[i,:]=mlbp.retrieval_modularities.loc[0,['retrieval_modularity','AMI_layer_avg','Accuracy_layer_avg']].values

    return [ep,eta,resgamma,omega]+list(allstats.mean())


def create_parallel_arguments(*args):

    #zips all of the arguments into tuples with all combinations from iterable types
    toproduct=[]
    for i,arg in enumerate(args):
        if hasattr(arg,'__iter__'):
            toproduct.append(arg)
        else:
            toproduct.append([arg])
    return list(itertools.product(*toproduct))


n=512
c=16.0
etas=np.linspace(0,1,50)
eps=np.linspace(0.01,1,50)
q=2
nlayers=40
resgamma=1.0
omega=1.0
ntrials=50
args=create_parallel_arguments(n,c,etas,eps,q,nlayers,resgamma,omega,ntrials)

print ("running on {:d} number of jobs".format(len(args)))
# res=_run_modbp_multilayer(args[0])
t=time()
output=pd.DataFrame(columns=['ep','eta','resgamma','omega','retrieval_modularity','AMI','Accuracy'])
with terminating(mp.Pool(processes=20)) as pool:
    res=pool.map(_run_modbp_multilayer,args)

for i,res_i in enumerate(res):
    output.loc[i,:]=res_i

outfile='eta_eps_scan_{:d}trials_{:d}nodes_{:d}layers.df.gz'.format(ntrials,n,nlayers)
with gzip.open(outfile,'w') as fh:
    pickle.dump(output,fh)

print("{:.4f} hours to run {:d} jobs".format((time()-t)/3600.0,len(args)))




