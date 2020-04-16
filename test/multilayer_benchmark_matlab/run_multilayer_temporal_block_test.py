from __future__ import division
import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import gzip,pickle
import matplotlib.pyplot as plt
import sys
from subprocess import Popen,PIPE
import re
import os
import scipy.io as scio
from  sklearn.cluster import KMeans
import scipy.sparse.linalg as slinalg
#generative multilayer benchmark models (now in python)
from time import time

from create_multiplex_functions import create_temporal_graph_block,call_gen_louvain,get_starting_partition_modularity,get_starting_partition_multimodbp_nodes
clusterdir=os.path.abspath('../..') # should be in test/multilayer_benchmark_matlab
matlabbench_dir=os.path.join(clusterdir, 'test/multilayer_benchmark_matlab/')
matlaboutdir = os.path.join(matlabbench_dir,"matlab_temp_outfiles")

if not os.path.exists(matlaboutdir):
    os.makedirs(matlaboutdir)
#main file for alling matlab

#shell scripts for calling matlab functions from command line
call_genlouvain_file = os.path.join(clusterdir,"test/genlouvain_mlsbm/call_matlab_genlouvain.sh")
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




#python run_multilayer_matlab_test.py

def run_louvain_multiplex_test(n,nlayers,mu,p_eta,omega,gamma,ntrials):
    ncoms=5

    finoutdir = os.path.join(matlabbench_dir, 'temporal_block_matlab_test_data_n{:d}_nlayers{:d}_trials{:d}_{:d}ncoms_multilayer'.format(n,nlayers,ntrials,ncoms))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)

    output = pd.DataFrame()
    outfile="{:}/temporal_block_test_n{:d}_L{:d}_mu{:.4f}_p{:.4f}_gamma{:.4f}_omega{:.4f}_trials{:d}.csv".format(finoutdir,n,nlayers,mu,p_eta, gamma,omega,ntrials)

    qmax=8
    max_iters=400
    print('running {:d} trials at gamma={:.4f}, omega={:.3f}, p={:.4f}, and mu={:.4f}'.format(ntrials,gamma,omega,p_eta,mu))
    for trial in range(ntrials):

        t=time()
        graph=create_temporal_graph_block(n_nodes=n, mu=mu,p_in=p_eta,p_out=0,n_blocks=1,
                                     n_layers=nlayers, ncoms=ncoms)
        graph.reorder_nodes()

        # plt.close()
        # f,a=plt.subplots(1,1,figsize=(8,8))
        # graph.plot_communities(ax=a)
        # plt.show()
        # with gzip.open("working_graph.gz",'wb') as fh:
        #     pickle.dump(graph,fh)
        # #
        # with gzip.open("working_graph.gz",'rb') as fh:
        #     graph=pickle.load(fh)

        print('time creating graph: {:.3f}'.format(time()-t))
        # start_vec = get_starting_partition_multimodbp_nodes(graph, gamma=gamma, omega=omega, q=ncoms)
        start_vec = get_starting_partition(graph, gamma=gamma, omega=10, q=ncoms)
        # print('time creating starting vec:{:.3f}'.format(time() - t))
        print('AMI start_vec', graph.get_AMI_with_communities(start_vec))
        ground_margs = create_marginals_from_comvec(start_vec, SNR=5,
                                                    q=qmax)

        mlbp = modbp.ModularityBP(mlgraph=graph, accuracy_off=True, use_effective=True,
                                  align_communities_across_layers_temporal=False, comm_vec=graph.comm_vec)
        bstars = [mlbp.get_bstar(q,omega=omega) for q in range(1, qmax+2,2)]
        # bstars = np.linspace(1,4,10)

        # bstars = [mlbp.get_bstar(qmax) ]

        #betas = np.linspace(bstars[0], bstars[-1], len(bstars) * 8)
        betas=bstars
        # betas=[.84]
        notconverged = 0
        for j,beta in enumerate(betas):
            t=time()
            mlbp.run_modbp(beta=beta, niter=max_iters, reset=True,
                           q=qmax,
                           starting_marginals=ground_margs,
                           resgamma=gamma, omega=omega)
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

            print(output.loc[cind,['beta','niters','AMI','AMI_layer_avg']])

            if trial == 0:  # write out whole thing
                with open(outfile, 'w') as fh:
                    output.to_csv(fh, header=True)
            else:
                with open(outfile, 'a') as fh:  # writeout last 2 rows for genlouvain + multimodbp
                    output.iloc[-1:, :].to_csv(fh, header=False)

            # if notconverged>1: #hasn't converged twice now.
            #     break
        #we now only call this once each trial with iterated version
        t=time()
        try:  # the matlab call has been dicey on the cluster for some.  This results in jobs quitting prematurely.
            S,t = call_gen_louvain(graph, gamma, omega)
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
            # print('genlouvain: {:.3f}'.format(ami_layer))
            output.loc[cind, 'NMI'] = nmi
            output.loc[cind, 'NMI_layer_avg'] = nmi_layer
            output.loc[cind, 'retrieval_modularity'] = cmod
            output.loc[cind, 'resgamma'] = gamma
            output.loc[cind, 'omega'] = omega
            Scoms, Scnt = np.unique(S, return_counts=True)
            output.loc[cind, 'num_coms'] = np.sum(Scnt > 5)
            matlabfailed = False
        except:
            e = sys.exc_info()[0]
            print("Error: {:}" .format( e))
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
    # plt.close()
    # f,a=plt.subplots(1,1,figsize=(5,5))
    # a.scatter(output['beta'].values,output['niters'].values)
    # a2=a.twinx()
    # a2.scatter(output['beta'].values,output['AMI_layer_avg'].values)
    #
    # plt.show()
    return 0


def main():
    n = int(sys.argv[1])  # nodes per layer
    nlayers = int(sys.argv[2])
    mu = float(sys.argv[3])
    p_eta = float(sys.argv[4])
    omega = float(sys.argv[5])
    gamma = float(sys.argv[6])
    ntrials = int(sys.argv[7])
    run_louvain_multiplex_test(n=n,nlayers=nlayers,mu=mu,p_eta=p_eta,omega=omega,gamma=gamma,
                               ntrials=ntrials)
    # run_louvain_multiplex_test(n=150,nlayers=100,mu=.7,p_eta=1.0,omega=4,gamma=1.0,ntrials=1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
