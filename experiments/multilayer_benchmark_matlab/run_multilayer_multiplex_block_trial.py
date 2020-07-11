from __future__ import division
import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import matplotlib.pyplot as plt
import sys
from subprocess import Popen,PIPE
import re
import os
import shutil
import gzip,pickle
import scipy.io as scio
import sklearn.metrics as skm
from  sklearn.cluster import KMeans
import scipy.sparse.linalg as slinalg
import itertools
#generative multilayer benchmark models (now in python)
import multilayerGM as gm
from time import time

from create_multiplex_functions import create_multiplex_graph
from create_multiplex_functions import create_multiplex_graph_matlab
clusterdir=os.path.abspath('../..') # should be in experiments/multilayer_benchmark_matlab
matlabbench_dir=os.path.join(clusterdir, 'experiments/multilayer_benchmark_matlab/')
matlaboutdir = os.path.join(matlabbench_dir,"matlab_temp_outfiles")

if not os.path.exists(matlaboutdir):
    os.makedirs(matlaboutdir)
#main file for alling matlab

#shell scripts for calling matlab functions from command line
call_genlouvain_file = os.path.join(clusterdir,"experiments/genlouvain_mlsbm/call_matlab_genlouvain.sh")
call_matlab_createbenchmark_file = os.path.join(matlabbench_dir, "call_matlab_multilayer.sh")

#set architecture flag for compiled files
oncluster=False
if re.search("/nas/longleaf",clusterdir):
    oncluster=True
arch = "elf64" if oncluster else "x86_64" #for different compiled code to run

def create_marginals_from_comvec(commvec,q=None,SNR=1000):
    if q is None:
        q=len(np.unique(commvec))

    outmargs=np.zeros((len(commvec),q))
    for i in range(len(commvec)):
        currow=np.array([1 for _ in range(q)])
        currow[int(commvec[i])]=SNR
        currow=1/np.sum(currow)*currow
        outmargs[i,:]=currow
    return outmargs

def get_starting_partition(mgraph,gamma=1.0,omega=1.0,q=2):
    """Spectral clustering on B matrix to initialize"""
    A, C = mgraph.to_scipy_csr()
    A+=A.T
    C+=C.T
    P = mgraph.create_null_adj()
    B=A - gamma*P  + omega*C
    evals, evecs = slinalg.eigs(B,k=q-1,which='LR')
    evecs=np.array(evecs)
    evecs2plot = np.real(evecs[:, np.flip(np.argsort(evals))])

    if q==2:
        mvec=(evecs2plot[:,0]>0).astype(int)
        return np.array(mvec).flatten()
    else:
        kmeans = KMeans(n_clusters=q, random_state=0).fit(evecs2plot)
        return kmeans.labels_


def call_gen_louvain(mgraph, gamma, omega, S=None):
    A, C = mgraph.to_scipy_csr()
    P = mgraph.create_null_adj()

    rprefix = np.random.randint(100000)
    scio_outfile = os.path.join(matlaboutdir, "{:d}_temp_matlab_input_file.mat".format(rprefix))
    matlaboutput = os.path.join(matlaboutdir, "{:d}_temp_matlab_output_file.mat".format(rprefix))
    T=mgraph.nlayers
    if S is None:
        scio.savemat(scio_outfile, {"A": A, "C": C, "P": P,"T":T})
    else:

        scio.savemat(scio_outfile, {"A": A, "C": C, "P": P,"T":T,
                                    "S0": np.reshape(S, (-1, mgraph.nlayers)).astype(float)})  # add in starting vector
    parameters = [call_genlouvain_file,
                  scio_outfile,
                  matlaboutput,
                  "{:.4f}".format(gamma),
                  "{:.4f}".format(omega)
                  ]
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    if process.returncode != 0:
        print("matlab call failed")
    print(stderr)

    try:
        S = scio.loadmat(matlaboutput)['S'][:, 0]
    except:
        print(stderr)
        os.remove(scio_outfile)
        raise (AssertionError,"matlab failed to run. can't find output file") #this should still in intercepted below
    ami = mgraph.get_AMI_with_communities(S)

    try:
        os.remove(scio_outfile)
    except:
        pass
    try:
        os.remove(matlaboutput)
    except:
        pass

    return S



#python run_multilayer_matlab_test.py

def run_louvain_multiplex_test(n,nlayers,mu,p_eta,omega,gamma,ntrials):
    ncoms=10

    finoutdir = os.path.join(matlabbench_dir, 'initialized_multiplex_block_matlab_test_data_n{:d}_nlayers{:d}_trials{:d}_{:d}ncoms_multilayer'.format(n,nlayers,ntrials,ncoms))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)

    output = pd.DataFrame()
    outfile="{:}/multiplex_block_test_n{:d}_L{:d}_mu{:.4f}_p{:.4f}_gamma{:.4f}_omega{:.4f}_trials{:d}.csv".format(finoutdir,n,nlayers,mu,p_eta, gamma,omega,ntrials)

    qmax=12
    max_iters=400
    print('running {:d} trials at gamma={:.4f}, omega={:.3f}, p={:.4f}, and mu={:.4f}'.format(ntrials,gamma,omega,p_eta,mu))
    for trial in range(ntrials):

        t=time()
        graph=create_multiplex_graph_matlab(n_nodes=n, mu=mu, p_in=p_eta,
                                           p_out=0,nblocks=3,nlayers=n_layers, ncoms=ncoms,ismultiplex=True)
        print('time creating graph: {:.3f}'.format(time()-t))

        start_vec = get_starting_partition(graph, gamma=gamma, omega=omega, q=ncoms)
        print('time creating starting vec:{:.3f}'.format(time() - t))
        print('AMI start_vec', graph.get_AMI_with_communities(start_vec))
        ground_margs = create_marginals_from_comvec(start_vec, SNR=5,
                                                    q=qmax)
        mlbp = modbp.ModularityBP(mlgraph=graph, accuracy_off=True, use_effective=True,
                                  align_communities_across_layers_multiplex=True, comm_vec=graph.comm_vec)
        bstars = [mlbp.get_bstar(q,omega=omega) for q in range(1, qmax+2,2)]
        # bstars = [mlbp.get_bstar(qmax) ]
        #betas = np.linspace(bstars[0], bstars[-1], len(bstars) * 8)
        betas=bstars
        notconverged = 0
        for j,beta in enumerate(betas):
            t=time()
            mlbp.run_modbp(beta=beta, niter=max_iters,
                           starting_marginals=ground_margs,
                           reset=False,
                           q=qmax, resgamma=gamma, omega=omega)
            print("time running modbp at mu,p={:.3f},{:.3f}: {:.3f}. niters={:.3f}".format(mu,p_eta,time()-t,mlbp.retrieval_modularities.iloc[-1,:]['niters']))
            mlbp_rm = mlbp.retrieval_modularities
            if mlbp_rm.iloc[-1,:]['converged'] == False: #keep track of how many converges we have
                notconverged+=1
            cind = output.shape[0]
            ind = mlbp_rm.index[mlbp_rm.shape[0] - 1]  # get last line
            for col in mlbp_rm.columns:
                output.loc[cind, col] = mlbp_rm.loc[ind, col]
            output.loc[cind, 'isGenLouvain'] = False
            output.loc[cind, 'mu'] = mu
            output.loc[cind, 'p'] = p_eta
            output.loc[cind, 'trial'] = trial

            # run genlouvain on graph
            t=time()


            if trial == 0:  # write out whole thing
                with open(outfile, 'w') as fh:
                    output.to_csv(fh, header=True)
            else:
                with open(outfile, 'a') as fh:  # writeout last 2 rows for genlouvain + multimodbp
                    output.iloc[-1:, :].to_csv(fh, header=False)

            if notconverged>1: #hasn't converged twice now.
                break
        #we now only call this once each trial with iterated version
        t=time()
        try:  # the matlab call has been dicey on the cluster for some.  This results in jobs quitting prematurely.
            S = call_gen_louvain(graph, gamma, omega)
            ami_layer = graph.get_AMI_layer_avg_with_communities(S)
            ami = graph.get_AMI_with_communities(S)
            nmi =  graph.get_AMI_with_communities(S,useNMI=True)
            nmi_layer  =  graph.get_AMI_layer_avg_with_communities(S,useNMI=True)

            cmod = modbp.calc_modularity(graph, S, resgamma=gamma, omega=omega)
            cind = output.shape[0]
            output.loc[cind, 'isGenLouvain'] = True
            output.loc[cind, 'mu'] = mu
            output.loc[cind, 'p'] = p_eta
            output.loc[cind, 'trial'] = trial
            output.loc[cind, 'AMI'] = ami
            output.loc[cind, 'AMI_layer_avg'] = ami_layer
            output.loc[cind, 'NMI'] = nmi
            output.loc[cind, 'NMI_layer_avg'] = nmi_layer
            output.loc[cind, 'retrieval_modularity'] = cmod
            output.loc[cind, 'resgamma'] = gamma
            output.loc[cind, 'omega'] = omega
            Scoms, Scnt = np.unique(S, return_counts=True)
            output.loc[cind, 'num_coms'] = np.sum(Scnt > 5)
            matlabfailed = False
        except:
            matlabfailed = True

        if not matlabfailed:
            with open(outfile, 'a') as fh:  # writeout last 2 rows for genlouvain + multimodbp
                output.iloc[-1:, :].to_csv(fh, header=False)

        print("time running matlab:{:.3f}. sucess: {:}".format(time() - t, str(not matlabfailed)))
        # if trial == 0:
        #     with open(outfile, 'w') as fh:
        #         output.to_csv(fh, header=True)
        # else:
        #     with open(outfile, 'a') as fh:  # writeout as we go
        #         output.iloc[[-1], :].to_csv(fh, header=False)

    return 0


def main():
    n = int(sys.argv[1])  # nodes per layer
    nlayers = int(sys.argv[2])
    mu = float(sys.argv[3])
    p_eta = float(sys.argv[4])
    omega = float(sys.argv[5])
    gamma = float(sys.argv[6])
    ntrials = int(sys.argv[7])
    run_louvain_multiplex_test(n=n,nlayers=nlayers,mu=mu,p_eta=p_eta,omega=omega,gamma=gamma,ntrials=ntrials)
    # run_louvain_multiplex_test(n=200,nlayers=15,mu=.8,p_eta=.5,omega=.5,gamma=1.0,ntrials=3)

    return 0

if __name__ == "__main__":
    sys.exit(main())
