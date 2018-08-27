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
import scipy.io as scio
import sklearn.metrics as skm
import itertools

clusterdir="/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/multilayer_benchmark_matlab"
#arch = "elf64"

#clusterdir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/test/multilayer_benchmark_matlab" #for testing locally

#clusterdir = "/Users/ben/Research (Github)/ModularityBP_Cpp/"
# finoutdir=os.path.join(clusterdir,'test/modbpdata/LFR_test_data_gamma3_beta2')

matlaboutdir = os.path.join(clusterdir,"/matlab_temp_files")
call_matlabfile = os.path.join(clusterdir,"call_matlab_multilayer.sh")

# def edges_to_adj(elist):
#     elist=np.array(elist)
#     max=np.max(elist)
#     A=np.zeros((max,max))
#     for i in range

def adjacency_to_edges(A,offset=0):
    nnz_inds = np.nonzero(A)
    nnzvals = np.array(A[nnz_inds])
    if len(nnzvals.shape) > 1:
        nnzvals = nnzvals[0]  # handle scipy sparse types
    return zip(nnz_inds[0]+offset, nnz_inds[1]+offset, nnzvals)

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



def create_multiplex_graph(n=1000,nlayers=40, ep=.99,eta=.1, c=10, mk=20, use_gcc=True,orig=None,layers=None, ismultiplex = False,ncoms=2):
    rprefix=np.random.randint(100000)
    rprefix_dir=os.path.join(clusterdir,str(rprefix))
    if not os.path.exists(rprefix_dir):
        os.makedirs(rprefix_dir)

    moutputfile=os.path.join(rprefix_dir,'network.mat')

    parameters = [call_matlabfile,
                  moutputfile,
                  "{:d}".format(n),
                  "{:d}".format(nlayers),
                  "{:.5f}".format(ep),
                  "{:.5f}".format(eta), #eta is the prop of transmitting community label!
                  "{:d}".format(ncoms)
                  ]
    print(parameters)
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    mlgraph=create_ml_graph_from_matlab(moutputfile,ismultiplex=ismultiplex)
    # for layer in mlgraph.layers:
    #     print ("k={:.4f}".format(2.0*layer.ecount()/layer.vcount()))
    # plt.close()
    # f,a=plt.subplots(1,1,figsize=(6,6))
    # mlgraph.plot_communities(ax=a)
    # plt.show()

    if os.path.exists("{:}".format(rprefix_dir)):
        shutil.rmtree("{:}".format(rprefix_dir))

    return mlgraph



# # run SBMBP on the input graph with the chosen q, using the EM algorithm to learn parameters
# # returns the AMI of the learned partition
# def run_SBMBP_on_graph(graph):
#     sbmbpfile = os.path.join(clusterdir,'test/mode_net/sbm')
#     # outdir = os.path.join(clusterdir,'test/modbpdata/LFR_test_data/')
#     rprefix = np.random.randint(100000)
#     tmp_grph_file = os.path.join(finoutdir, '{:d}temporary_graph_file.gml'.format(rprefix))
#     graph.save(tmp_grph_file)
#     all_partitions = {}
#     final_values = {}
#     for q in range(2, 5):
#         parameters = [
#             sbmbpfile, 'learn',
#             "-l", tmp_grph_file,
#             '-q', '{:d}'.format(q),
#             '-M', '{:}_q{:d}_marginals.txt'.format(tmp_grph_file, q),
#             '-d', '1',
#             '-i', '1'
#             #         '-L','{:}_q{:d}_planted_cab.txt'.format(grph_file,q),
#             #         '--spcmode','{:d}'.format(0),
#             #         '--wcab','{:}_q{:d}_cab.txt'.format(grph_file,q)
#         ]
#         process = Popen(parameters, stderr=PIPE, stdout=PIPE)
#         stdout, stderr = process.communicate()
#         if process.returncode != 0:
#             raise RuntimeError("running SBMBP failed : {:}".format(stderr))
#         # print(stdout)
#         marginal_file = '{:}_q{:d}_marginals.txt'.format(tmp_grph_file, q)
#         marginals = []
#         partition = []
#         inmargs = False
#         inpartition = False
#         with open(marginal_file, 'r') as f:
#
#             for i, line in enumerate(f.readlines()):
#                 if re.search("\A\s*\Z", line):  # only while space
#                     continue
#                 if i == 0:
#                     fin_vals = dict([tuple(val.split('=')) for val in line.split()])
#                     for k, val in fin_vals.items():
#                         fin_vals[k] = float(val)
#                     final_values[q] = fin_vals
#                 if re.search('marginals:', line):
#                     inmargs = True
#                     inpartition = False
#                     continue
#                 if re.search('argmax_configuration', line):
#                     inmargs = False
#                     inpartition = True
#                     continue
#                 if inmargs:
#                     marginals.append(line.split())
#                 if inpartition:
#                     partition = line.split()
#
#         partition = np.array(partition, dtype=int)
#         all_partitions[q] = partition
#         if os.path.exists(marginal_file):
#             os.remove(marginal_file)
#     if os.path.exists(tmp_grph_file):
#         os.remove(tmp_grph_file)
#
#
#     minq = sorted(final_values.items(), key=lambda x: x[1]['f'])[0][0]
#
#     AMI=skm.adjusted_mutual_info_score(all_partitions[q], graph.vs['block'])
#     return AMI

#python run_LFR_test_with_sbmbp.py 100 .1 4 1.0 1 2 1.0
def main():
    n = int(sys.argv[1])
    c = float(sys.argv[2])
    nlayers=int(sys.argv[3])
    ep = float(sys.argv[4])
    eta= float(sys.argv[5])
    omega=float(sys.argv[6])
    gamma = float(sys.argv[7])
    ntrials= int(sys.argv[8])
    ncoms=4

    finoutdir = os.path.join(clusterdir, 'multiplex_matlab_test_data_n{:d}_nlayers{:d}_trials{:d}_k{:.2f}_{:d}ncoms_multilayer'.format(n,nlayers,ntrials,c,ncoms))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)

    output=pd.DataFrame(columns=['ep','beta', 'resgamma', 'niters', 'AMI','retrieval_modularity','isSBM'])
    outfile="{:}/multiplex_test_n{:d}_L{:d}_eps{:.4f}_eta{:.4f}_gamma{:.4f}_omega{:.4f}_trials{:d}.csv".format(finoutdir,n,nlayers,ep,eta, gamma,omega,ntrials)


    qmax=8
    max_iters=4000
    print('running {:d} trials at gamma={:.4f} and eps={:.4f}'.format(ntrials,gamma,ep))
    for trial in range(ntrials):

        graph=create_multiplex_graph(n=n, ep=ep, eta=eta, c=c, mk=20, use_gcc=True,
                                     nlayers=nlayers,ismultiplex=True,ncoms=ncoms)
        #graph.layers[0].save('test_LFR_onelayer.graphml.gz')
        # ami_sbm=run_SBMBP_on_graph(graph)
        # cind = output.shape[0]
        # output.loc[cind,['beta','resgamma','niters','retrieval_modularity']]=[None,None,None,None]
        # output.loc[cind,'AMI']=ami_sbm
        # output.loc[cind,'isSBM']=True
        # output.loc[cind,'ep']=ep
        mlbp = modbp.ModularityBP(mlgraph=graph,accuracy_off=True,use_effective=True,
                                  comm_vec=graph.comm_vec)
        bstars = [mlbp.get_bstar(q) for q in range(2, qmax)]
        #betas = np.linspace(bstars[0], bstars[-1], len(bstars) * 8)
        betas=bstars
        for beta in betas:
            mlbp.run_modbp(beta=beta, niter=max_iters, q=qmax, resgamma=gamma, omega=omega)
            mlbp_rm = mlbp.retrieval_modularities
            print ("AMI:", mlbp_rm.loc[mlbp.nruns - 1, 'AMI'])
            print ("AMI:", mlbp_rm.loc[mlbp.nruns - 1, 'converged'])
            print ("AMI:", mlbp_rm.loc[mlbp.nruns - 1, 'niters'])


            print(beta)

            # for debugging
            # plt.close()
            # f,a=plt.subplots(1,2,figsize=(8,4))
            # a=plt.subplot(1,2,1)
            # a.set_title("ground communities")
            # mlbp.plot_communities(ax=a)
            # a = plt.subplot(1, 2, 2)
            # a.set_title("discovered")
            # mlbp.plot_communities(ax=a,ind=mlbp.nruns-1)
            # plt.show()


        # ind2keep=np.where(np.logical_and(mlbp_rm['converged'],~mlbp_rm['is_trivial']))[0]
        ind2keep=np.where(mlbp_rm['converged'])[0]#we switched to convergence as only criteria

        cind = output.shape[0]
        if len(ind2keep)>0:
            minidx = mlbp_rm.iloc[ind2keep]['retrieval_modularity'].idxmax()
            for col in mlbp_rm.columns.values:
                output.loc[cind,col]=mlbp_rm.loc[minidx,col]
        else:

            for col in mlbp_rm.columns.values:
                #just take first one to get run information
                output.loc[cind,col]=mlbp_rm.iloc[0][col]
            output.loc[cind,'converged']=False
            output.loc[cind,'niters']=max_iters+1

        output.loc[cind,'ep']=ep
        output.loc[cind,'eta']=eta

        if trial == 0:
            with open(outfile, 'w') as fh:
                output.to_csv(fh, header=True)
        else:
            with open(outfile, 'a') as fh:  # writeout as we go
                output.iloc[[-1], :].to_csv(fh, header=False)

    return 0

if __name__ == "__main__":
    #create_lfr_graph(n=1000, ep=.1, c=4, mk=12, use_gcc=True,orig=2,layers=2, multiplex = True)
    main()
