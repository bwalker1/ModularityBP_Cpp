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
import itertools
#generative multilayer benchmark models (now in python)
import multilayerGM as gm
from time import time
from simple_pid import PID

from create_multiplex_functions import create_multiplex_graph
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

    finoutdir = os.path.join(matlabbench_dir, 'sandboxing_multilayer_test')
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)

    output = pd.DataFrame()
    outfile="{:}/sandboxing_test_n{:d}_L{:d}_mu{:.4f}_p{:.4f}_gamma{:.4f}_omega{:.4f}_trials{:d}.csv".format(finoutdir,n,nlayers,mu,p_eta, gamma,omega,ntrials)

    qmax=12
    max_iters=1000
    print('running {:d} trials at gamma={:.4f}, omega={:.3f}, p={:.4f}, and mu={:.4f}'.format(ntrials,gamma,omega,p_eta,mu))
    for trial in range(ntrials):

        t=time()

        load =True
        if not load:
            graph=create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                          n_layers=nlayers, ncoms=ncoms)
            with gzip.open("working_graph.gz",'wb') as fh:
                pickle.dump(graph,fh)
        else:
            with gzip.open("working_graph.gz",'rb') as fh:
              graph=pickle.load(fh)

        print('time creating graph: {:.3f}'.format(time()-t))
        mlbp = modbp.ModularityBP(mlgraph=graph, accuracy_off=True, use_effective=False,
                                  normalize_edge_weights=False,
                                  align_communities_across_layers_multiplex=False,
                                  align_communities_across_layers_temporal=False,
                                  comm_vec=graph.comm_vec)

        bstars = [mlbp.get_bstar(q,omega=omega) for q in range(3, qmax+2,1)]
        print('bstars',bstars)
        # bstars = np.linspace(1,4,10)

        # bstars = [mlbp.get_bstar(qmax) ]

        #betas = np.linspace(bstars[0], bstars[-1], len(bstars) * 8)
        betas=bstars
        betas=[1.25]
        # betas=[ mlbp.get_bstar(q=qmax,omega=omega)]

        notconverged = 0
        for j,beta in enumerate(betas):
            print("beta: {:.3f}".format(beta))
            t=time()


            #create and reset bp object
            mlbp.run_modbp(beta=beta, niter=1000, reset=True,dumping_rate=0.0,
                           normalize_edge_weights=False,
                           q=qmax, resgamma=gamma, omega=omega,anneal_omega=True)

            # mlbp._bpmod.setBeta(beta)# resets
            # mlbp._bpmod.setq(qmax)
            # print('q',mlbp._bpmod.getq())

            # pid = PID(1, 0.1, 0.05,sample_time=0,output_limits=[-.02,-.001], setpoint=.2)

            # assume we have a system we want to control in controlled_system

            # dr=.001
            #
            # drs=[.02,.05,.1,.2,.4,1]
            # # drs=[1]
            # interval=100
            #
            # for k in range(max_iters):
            #     # dr=np.min([.1,k/max_iters])
            #
            #     dr=drs[np.min([k//interval,len(drs)-1])]
            #     if k%5==0:
            #         cmargs = np.array(mlbp._bpmod.return_marginals())
            #         centrop = modbp._get_avg_entropy(cmargs)
            #         mlbp.marginals[0]=cmargs
            #         cpart=mlbp._get_partition(0,use_effective=True)
            #         mlbp.partitions[0]=cpart
            #         cami=graph.get_AMI_layer_avg_with_communities(cpart)
            #         cmod=modbp.calc_modularity(graph,cpart,resgamma=gamma,omega=omega)
            #         # print('q',mlbp._bpmod.getq())
            #         _,cnts=np.unique(cpart,return_counts=True)
            #         print('iters: {:d}, dr: {:.3f}, entropy : {:.3f}, ncoms:{:d}, AMI: {:.4f}, mod: {:.3f}, cnts:{:}'.format(k,dr,centrop,len(np.unique(cpart)),cami,cmod,str(cnts)))
            #         if k>0 and k%interval==0: #perform alignment across layers
            #             # mlbp._get_community_distances(0 ,use_effective=True)
            #             cpart = mlbp._get_partition(0, use_effective=False)
            #             mlbp.partitions[0] = cpart
            #             # mlbp._merge_communities_bp(0)
            #             nsweeps=mlbp._perform_permuation_sweep_multiplex(0)
            #             mlbp._switch_beliefs_bp(0)
            #
            #     # dr = -1*pid(centrop)
            #     mlbp._bpmod.setDumpingRate(dr)
            #     changed=mlbp._bpmod.step()
            #     if changed==False:
            #         print('no change')
            #         break
            #
            # # print("time running modbp at mu,p={:.3f},{:.3f}: {:.3f}. niters={:.3f}".format(mu,p_eta,time()-t,mlbp.retrieval_modularities.iloc[-1,:]['niters']))
            # mlbp_rm = mlbp.retrieval_modularities
            # if mlbp_rm.iloc[-1,:]['converged'] == False: #keep track of how many converges we have
            #     notconverged+=1
            # cind = output.shape[0]
            # ind = mlbp_rm.index[mlbp_rm.shape[0] - 1]  # get last line
            # for col in mlbp_rm.columns:
            #     output.loc[cind, col] = mlbp_rm.loc[ind, col]
            # output.loc[cind, 'isGenLouvain'] = False
            # output.loc[cind, 'mu'] = mu
            # output.loc[cind, 'p'] = p_eta
            # output.loc[cind, 'trial'] = trial
            #
            # # run genlouvain on graph
            # t=time()

    #         print(output.loc[cind,['beta','niters','AMI','AMI_layer_avg']])
    #
    #         if trial == 0:  # write out whole thing
    #             with open(outfile, 'w') as fh:
    #                 output.to_csv(fh, header=True)
    #         else:
    #             with open(outfile, 'a') as fh:  # writeout last 2 rows for genlouvain + multimodbp
    #                 output.iloc[-1:, :].to_csv(fh, header=False)
    #
    #         # if notconverged>1: #hasn't converged twice now.
    #         #     break
    #     #we now only call this once each trial with iterated version
    #     t=time()
    #     try:  # the matlab call has been dicey on the cluster for some.  This results in jobs quitting prematurely.
    #         S = call_gen_louvain(graph, gamma, omega)
    #         ami_layer = graph.get_AMI_layer_avg_with_communities(S)
    #         ami = graph.get_AMI_with_communities(S)
    #         nmi =  graph.get_AMI_with_communities(S,useNMI=True)
    #         nmi_layer  =  graph.get_AMI_layer_avg_with_communities(S,useNMI=True)
    #
    #         cmod = modbp.calc_modularity(graph, S, resgamma=gamma, omega=omega)
    #         cind = output.shape[0]
    #         output.loc[cind, 'isGenLouvain'] = True
    #         output.loc[cind, 'mu'] = mu
    #         output.loc[cind, 'p'] = p_eta
    #         output.loc[cind, 'trial'] = trial
    #         output.loc[cind, 'AMI'] = ami
    #         output.loc[cind, 'AMI_layer_avg'] = ami_layer
    #         output.loc[cind, 'NMI'] = nmi
    #         output.loc[cind, 'NMI_layer_avg'] = nmi_layer
    #         output.loc[cind, 'retrieval_modularity'] = cmod
    #         output.loc[cind, 'resgamma'] = gamma
    #         output.loc[cind, 'omega'] = omega
    #         Scoms, Scnt = np.unique(S, return_counts=True)
    #         output.loc[cind, 'num_coms'] = np.sum(Scnt > 5)
    #         matlabfailed = False
    #     except:
    #         matlabfailed = True
    #
    #     if not matlabfailed:
    #         with open(outfile, 'a') as fh:  # writeout last 2 rows for genlouvain + multimodbp
    #             output.iloc[-1:, :].to_csv(fh, header=False)
    #
    #     print("time running matlab:{:.3f}. sucess: {:}".format(time() - t, str(not matlabfailed)))
    #     # if trial == 0:
    #     #     with open(outfile, 'w') as fh:
    #     #         output.to_csv(fh, header=True)
    #     # else:
    #     #     with open(outfile, 'a') as fh:  # writeout as we go
    #     #         output.iloc[[-1], :].to_csv(fh, header=False)
    # plt.close()
    # f,a=plt.subplots(1,1,figsize=(5,5))
    # a.scatter(output['beta'].values,output['niters'].values)
    # a2=a.twinx()
    # a2.scatter(output['beta'].values,output['AMI_layer_avg'].values)

    # plt.show()
    return 0


def main():
    run_louvain_multiplex_test(n=1000,nlayers=15,mu=.9,p_eta=1.0,omega=.3,gamma=1.0,ntrials=1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
