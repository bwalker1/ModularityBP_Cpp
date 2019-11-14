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
    if S is None:
        scio.savemat(scio_outfile, {"A": A, "C": C, "P": P})
    else:

        scio.savemat(scio_outfile, {"A": A, "C": C, "P": P,
                                    "S0": np.reshape(S, (-1, mgraph.nlayers)).astype(float)})  # add in starting vector
    print(call_genlouvain_file)
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

    S = scio.loadmat(matlaboutput)['S'][:, 0]
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

def adjacency_to_edges(A,offset=0):
    nnz_inds = np.nonzero(A)
    nnzvals = np.array(A[nnz_inds])
    if len(nnzvals.shape) > 1:
        nnzvals = nnzvals[0]  # handle scipy sparse types
    return list(zip(nnz_inds[0]+offset, nnz_inds[1]+offset, nnzvals))

def create_ml_graph_from_matlab(moutputfile,ismultiplex=True):
    matoutputdict=scio.loadmat(moutputfile)
    A=matoutputdict['A']
    nnodes=A[0][0].shape[0]
    nlayers=len(A)
    layer_vec=[ i//nnodes for i in range(nnodes*nlayers)]
    for i,A in enumerate(A):
        if i==0:
            all_intra_edges=adjacency_to_edges(A[0])
        else:
            all_intra_edges+=adjacency_to_edges(A[0],offset=i*nnodes)

    interlayer_edges=[]
    for i in range(nnodes):
        if ismultiplex:
            #connect all possible pairs state nodes
            interlayer_edges.extend( itertools.combinations([i+l*nnodes for l in range(nlayers)],2))
        else: #temporal case only connect adjacent
            interlayer_edges.extend([ (i+(l*nnodes), i+(l+1)*nnodes) for l in range(1,nlayers-1)])

    comm_vec=matoutputdict['S'].flatten('F') #flatten by column


    mlgraph=modbp.MultilayerGraph(intralayer_edges=all_intra_edges,interlayer_edges=interlayer_edges,\
                                  layer_vec=layer_vec,comm_vec=comm_vec)

    return mlgraph


def convert_nxmg_to_mbp_multigraph(nxmg, dt):
    # dt has the interlayer edges in it
    nodelist = np.array(list(nxmg.adj.keys()))
    layervec = nodelist[:, 1]
    N = len(nodelist)
    layers, layercounts = np.unique(layervec, return_counts=True)
    assert (len(np.unique(layercounts)) == 1), "Multiplex must have same number of edges in each layer"
    nodeperlayer = layercounts[0]

    layer_adjust_ind_dict = dict(zip(layers, np.append([0], np.cumsum(layercounts)[:-1])))
    node_inds = dict([((n, lay), layer_adjust_ind_dict[lay] + n) for n, lay in nodelist])
    interelist = []
    intraelist = []
    edges = np.array(nxmg.edges)
    # seperate edges by type
    for e1, e2 in edges:
        ind1 = node_inds[(e1[0], e1[1])]
        ind2 = node_inds[(e2[0], e2[1])]
        assert e1[1] == e2[1], "Non intralayer edges identified in multiplex"
        intraelist.append((ind1, ind2))

    # We create a multiplex interedge list here
    for i in range(nodeperlayer):  # i is node number
        curnodes = [i + j * (nodeperlayer) for j in range(len(layers))]
        for ind1, ind2 in itertools.combinations(curnodes, 2):
            interelist.append((ind1, ind2))

    partition = list(nxmg.nodes(data='mesoset'))

    partition = list(map(lambda x: (node_inds[x[0]], x[1]), partition))
    partition = sorted(partition, key=lambda x: x[0])
    comvec = [x[1] for x in partition]
    return modbp.MultilayerGraph(comm_vec=comvec, interlayer_edges=interelist,
                                 intralayer_edges=intraelist,
                                 layer_vec=layervec)


def create_multiplex_graph(n_nodes=100, n_layers=5, mu=.99, p=.1, maxcoms=10, k_max=150,
                           k_min=3):
    theta = 1
    dt = gm.dependency_tensors.UniformMultiplex(n_nodes, n_layers, p)
    null = gm.dirichlet_null(layers=dt.shape[1:], theta=theta, n_sets=maxcoms)
    partition = gm.sample_partition(dependency_tensor=dt, null_distribution=null)

    # with use the degree corrected SBM to mirror paper
    multinet = gm.multilayer_DCSBM_network(partition, mu=mu, k_min=k_min, k_max=k_max, t_k=2)
    #     return multinet
    mbpmulltinet = convert_nxmg_to_mbp_multigraph(multinet, dt)
    return mbpmulltinet


#original mehtod used the matlab code.  have since switched to the python .
def create_multiplex_graph_matlab(n=1000, nlayers=40, mu=.99, p=.1,
                            use_gcc=True, orig=None, layers=None, ismultiplex = False, ncoms=2):
    rprefix=np.random.randint(1000000)
    rprefix_dir=os.path.join(matlaboutdir,str(rprefix))
    if not os.path.exists(rprefix_dir):
        os.makedirs(rprefix_dir)

    moutputfile=os.path.join(rprefix_dir,'network.mat')

    parameters = [call_matlab_createbenchmark_file,
                  moutputfile,
                  "{:d}".format(n),
                  "{:d}".format(nlayers),
                  "{:.5f}".format(mu),
                  "{:.5f}".format(p),  #p is the prop of transmitting community label!
                  "{:d}".format(ncoms)
                  ]
    print(parameters)
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("creating benchmark graph failed : {:}".format(stderr))

    mlgraph=create_multiplex_graph(moutputfile,ismultiplex=ismultiplex)

    #clean out random graph
    if os.path.exists("{:}".format(rprefix_dir)):
        shutil.rmtree("{:}".format(rprefix_dir))

    return mlgraph


#python run_multilayer_matlab_test.py

def run_louvain_multiplex_test(n,nlayers,mu,p_eta,omega,gamma,ntrials):
    ncoms=10

    finoutdir = os.path.join(matlabbench_dir, 'multiplex_matlab_test_data_n{:d}_nlayers{:d}_trials{:d}_{:d}ncoms_multilayer'.format(n,nlayers,ntrials,ncoms))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)

    output = pd.DataFrame()
    outfile="{:}/multiplex_test_n{:d}_L{:d}_mu{:.4f}_p{:.4f}_gamma{:.4f}_omega{:.4f}_trials{:d}.csv".format(finoutdir,n,nlayers,mu,p_eta, gamma,omega,ntrials)

    qmax=12
    max_iters=4000
    print('running {:d} trials at gamma={:.4f}, omega={:.3f}, p={:.4f}, and mu={:.4f}'.format(ntrials,gamma,omega,p_eta,mu))
    for trial in range(ntrials):

        t=time()
        graph=create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                     n_layers=nlayers, maxcoms=ncoms)
        print('time creating graph: {:.3f}'.format(time()-t))
        with gzip.open("notworking_graph.gz",'wb') as fh:
            pickle.dump(graph,fh)
        mlbp = modbp.ModularityBP(mlgraph=graph,accuracy_off=True,use_effective=True,align_communities_across_layers=False,
                                  comm_vec=graph.comm_vec)
        bstars = [mlbp.get_bstar(q) for q in range(4, qmax+2,2)]
        # bstars = [mlbp.get_bstar(ncoms) ]

        #betas = np.linspace(bstars[0], bstars[-1], len(bstars) * 8)
        betas=bstars
        notconverged = 0
        for j,beta in enumerate(betas):
            t=time()
            mlbp.run_modbp(beta=beta, niter=max_iters, reset=False,
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
            output.loc[cind, 'trial'] = trial

            # run genlouvain on graph
            t=time()
            try:  # the matlab call has been dicey on the cluster for some.  This results in jobs quitting prematurely.

                if j == 0:
                    S = call_gen_louvain(graph, gamma, omega)
                else:
                    S = call_gen_louvain(graph, gamma, omega, S)  # use output from previous run

                ami_layer = graph.get_AMI_layer_avg_with_communities(S)
                ami = graph.get_AMI_with_communities(S)
                cmod = modbp.calc_modularity(graph, S, resgamma=gamma, omega=omega)
                cind = output.shape[0]
                output.loc[cind, 'isGenLouvain'] = True
                output.loc[cind, 'mu'] = mu
                output.loc[cind, 'trial'] = trial
                output.loc[cind, 'AMI'] = ami
                output.loc[cind, 'AMI_layer_avg'] = ami_layer
                output.loc[cind, 'retrieval_modularity'] = cmod
                output.loc[cind, 'resgamma'] = gamma
                output.loc[cind, 'omega'] = omega
                output.loc[cind, 'gl_iter_num'] = j
                Scoms, Scnt = np.unique(S, return_counts=True)
                output.loc[cind, 'num_coms'] = np.sum(Scnt > 5)

                matlabfailed = False
            except:
                matlabfailed = True
            print("time running matlab:{:.3f}. sucess: {:}".format(time()-t,str(not matlabfailed)))

            if trial == 0:  # write out whole thing
                with open(outfile, 'w') as fh:
                    output.to_csv(fh, header=True)
            else:
                row2write = 2 if not matlabfailed else 1
                with open(outfile, 'a') as fh:  # writeout last 2 rows for genlouvain + multimodbp
                    output.iloc[-row2write:, :].to_csv(fh, header=False)
            if notconverged>1: #hasn't converged twice now.
                break

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
    #run_louvain_multiplex_test(n=300,nlayers=5,mu=1.0,p_eta=.5,omega=.023,gamma=1.0,ntrials=1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
