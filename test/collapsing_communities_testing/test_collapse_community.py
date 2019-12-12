import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import modbp
import seaborn as sbn
import logging
import gzip,pickle
import os,sys,re
from time import time
sys.path.append(os.path.abspath("../multilayer_benchmark_matlab"))
from create_multiplex_functions import create_multiplex_graph
from create_multiplex_functions import create_multiplex_graph_matlab
from create_multiplex_functions import call_gen_louvain
logging.basicConfig(level=logging.ERROR)

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

def expand_marginals(MMgraph,marginals):
    col_map=MMgraph.collapse_map
    new_marginals=np.zeros((len(col_map),marginals.shape[1]))
    for k,val in col_map.items():
        new_marginals[k,:]=marginals[val,:]
    return new_marginals

def test_collapse():
    n=100
    nlayers=1
    ep=.2
    eta=.3
    c=8
    ncoms=3
    cmap=sbn.cubehelix_palette(as_cmap=True)
    graph=modbp.generate_planted_partitions_dynamic_sbm(n=n,epsilon=ep,eta=eta,c=c,
                                                        nlayers=nlayers,ncoms=ncoms)


    collapse_graph=modbp.convertMultilayertoMergedMultilayer(graph)
    collapse_graph=collapse_graph.createCollapsedGraph(graph.comm_vec)

    print("AMI",collapse_graph.get_AMI_with_communities(labels=np.array([0,1,2,])))



    A,C=graph.to_scipy_csr()
    A=np.array(A.toarray())
    C=np.array(C.toarray())

    Ac,Cc=collapse_graph.to_scipy_csr()
    Ac=np.array(Ac.toarray())
    Cc=np.array(Cc.toarray())

    inds1=np.where(collapse_graph.comm_vec==0)[0]
    inds2=np.where(collapse_graph.comm_vec==1)[0]

    c1c2=np.sum(A[np.ix_(inds1,inds2)])

    plt.close()
    f,a=plt.subplots(2,2,figsize=(8,8))
    a=plt.subplot(2,2,1)
    plt.pcolormesh(A,cmap=cmap)
    a=plt.subplot(2,2,2)
    plt.pcolormesh(C,cmap=cmap)
    a = plt.subplot(2, 2, 3)
    plt.pcolormesh(Ac, cmap=cmap)
    a = plt.subplot(2, 2, 4)
    plt.pcolormesh(Cc, cmap=cmap)

    plt.show()


def test_run_modbp_on_collapse():
    n = 200
    nlayers = 50
    ep = .5
    eta = 0
    c = 8
    ncoms = 3
    cmap = sbn.cubehelix_palette(as_cmap=True)
    graph = modbp.generate_planted_partitions_dynamic_sbm(n=n, epsilon=ep, eta=eta, c=c,
                                                          nlayers=nlayers, ncoms=ncoms)



    # rand_coms=np.random.choice(range(ncoms),size=n)
    # rand_coms=np.array([rand_coms for _ in range(nlayers)]).flatten()
    rand_coms=np.array([range(n) for _ in range(nlayers) ]).flatten()

    collapse_graph = modbp.convertMultilayertoMergedMultilayer(graph)
    collapse_graph = collapse_graph.createCollapsedGraph(rand_coms,maintain_sparsity=False)

    collapse_graph.normalize_edge_weights(omega=1.0)
    # print(list(zip(collapse_graph.intralayer_edges,collapse_graph.intralayer_weights)))

    # visualize collapsed graphs
    A, C = graph.to_scipy_csr()
    A = np.array(A.toarray())
    C = np.array(C.toarray())
    #
    Ac, Cc = collapse_graph.to_scipy_csr()
    Ac = np.array(Ac.toarray())
    Cc = np.array(Cc.toarray())
    #
    # inds1 = np.where(collapse_graph.comm_vec == 0)[0]
    # inds2 = np.where(collapse_graph.comm_vec == 1)[0]
    #
    # c1c2 = np.sum(A[np.ix_(inds1, inds2)])

    plt.close()
    f, a = plt.subplots(1, 1, figsize=(5, 5))
    # a = plt.subplot(2, 2, 1)
    # a.set_title("A uncollapsed")
    # # plt.pcolormesh(A, cmap=cmap)
    # a = plt.subplot(2, 2, 2)
    # a.set_title("C uncollapsed")
    # # plt.pcolormesh(C, cmap=cmap)
    a = plt.subplot(1, 1, 1)
    a.set_title("A collapsed")
    plt.pcolormesh(Ac, cmap=cmap)
    # a = plt.subplot(2, 2, 4)
    # a.set_title("C collapsed")
    # plt.pcolormesh(Cc, cmap=cmap)
    plt.show()

    # node_strengths=collapse_graph.get_intralayer_degrees(weighted=True)
    # print(node_strengths)

    bpobj=modbp.ModularityBP(mlgraph=collapse_graph,use_effective=False,
                             align_communities_across_layers_temporal=False,
                             align_communities_across_layers_multiplex=False)
    beta=bpobj.get_bstar(q=ncoms,omega=1.0)
    print("beta = {:.3f} ".format(beta))

    # for beta in np.linspace(.1,3.5,20):
    #     print("beta = {:.3f} ".format(beta))
    bpobj.run_modbp(beta=beta,niter=300,q=ncoms,
                    resgamma=1.0,omega=1.0,anneal_omega=False)

    rm_df = bpobj.retrieval_modularities
    print(rm_df.loc[:, ['AMI', "AMI_layer_avg"]])

    new_margs=expand_marginals(collapse_graph,bpobj.marginals[0])
    #
    bpobj2=modbp.ModularityBP(mlgraph=graph,use_effective=False,align_communities_across_layers_multiplex=False,
                              align_communities_across_layers_temporal=False)
    beta2=bpobj2.get_bstar(q=ncoms,omega=1.0)
    print("beta = {:.3f} ".format(beta2))

    bpobj2.run_modbp(beta=beta2,q=ncoms,niter=0,resgamma=1.0,omega=1.0)
    new_belief=bpobj2._create_beliefs_from_marginals(new_margs)
    bpobj2._set_beliefs(new_belief)
    bpobj2.run_modbp(beta=beta2,q=ncoms,niter=1000, resgamma=1.0, omega=1.0,reset=False)
    rm_df=bpobj2.retrieval_modularities

    print(rm_df.loc[:,['AMI',"AMI_layer_avg"]])

    return 0



def get_starting_partition(mgraph,gamma=1.0):

    A, C = mgraph.to_scipy_csr()
    A=np.array(A.toarray())
    A+=A.T

    P = mgraph.create_null_adj()
    B=A -  gamma*P

    evals, evecs = np.linalg.eig(B)
    evecs2plot = evecs[:, np.flip(np.argsort(evals))]

    mvec=(evecs2plot[:,0]>0).astype(int)


    # plt.close()
    # f,a=plt.subplots(1,2,figsize=(6,3))
    # x=np.real(evals)
    # y=np.imag(evals)
    # a[0].scatter(x,y)
    # a[1].hist(x,bins=35)
    # plt.show()

    return mvec



def test_on_multiplex_block():
    n = 1000
    nlayers = 15
    mu = .9
    p_eta = 1.0
    ncoms = 10
    omega = 1.0

    t = time()

    load = False
    if not load:
        # multipex = create_multiplex_graph_matlab(n_nodes=n, mu=mu, p_in=p_eta,nblocks=3,
        #                                          nlayers=nlayers, ncoms=ncoms)
        multipex = create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                                 n_layers=nlayers, ncoms=ncoms)
        with gzip.open("working_graph.gz", 'wb') as fh:
            pickle.dump(multipex, fh)
    else:
        with gzip.open("working_graph.gz", 'rb') as fh:
            multipex = pickle.load(fh)


    multipex.reorder_nodes()
    print(np.unique(multipex.comm_vec,return_counts=True))

    rand_coms = np.array([range(n) for _ in range(nlayers)]).flatten()
    # rand_coms = np.array(range(n*nlayers))

    collapse_graph = modbp.convertMultilayertoMergedMultilayer(multipex)
    collapse_graph = collapse_graph.createCollapsedGraph(rand_coms,maintain_sparsity=False)
    
    ig_col=collapse_graph._export_to_igraph()
    ig_col.save("working_graph.graphml.gz")
    # collapse_graph.normalize_edge_weights(omega=1.0)


    print('time creating graph: {:.3f}'.format(time() - t))


    # visualize collapsed graphs
    # A, C = multipex.to_scipy_csr()
    # A = np.array(A.toarray())
    # C = np.array(C.toarray())
    # #
    # Ac, Cc = collapse_graph.to_scipy_csr()
    #
    # Ac = np.array(Ac.toarray())
    # Cc = np.array(Cc.toarray())
    #
    # Atot=np.zeros(Ac.shape)
    # coms=np.unique(multipex.layer_vec)
    # for com in coms:
    #     cinds=np.where(multipex.layer_vec==com)[0]
    #     Atot+=(A[np.ix_(cinds, cinds)])
    #
    # cmap = sbn.cubehelix_palette(as_cmap=True)
    #
    # plt.close()
    # f, a = plt.subplots(1, 2, figsize=(8, 4))
    # # a = plt.subplot(2, 2, 1)
    # # a.set_title("A uncollapsed")
    # # # plt.pcolormesh(A, cmap=cmap)
    # # a = plt.subplot(2, 2, 2)
    # # a.set_title("C uncollapsed")
    # # # plt.pcolormesh(C, cmap=cmap)
    # a = plt.subplot(1, 2, 1)
    # a.set_title("A collapsed")
    # plt.pcolormesh(Ac, cmap=cmap)
    # a = plt.subplot(1, 2, 2)
    # a.set_title("A summed")
    # plt.pcolormesh(Atot, cmap=cmap)
    # # a = plt.subplot(2, 2, 4)
    # # a.set_title("C collapsed")
    # # plt.pcolormesh(Cc, cmap=cmap)
    # plt.show()

    qmax=ncoms+2

    bpobj=modbp.ModularityBP(mlgraph=collapse_graph,use_effective=False,
                             align_communities_across_layers_temporal=False,
                             align_communities_across_layers_multiplex=False)

    betas=[ bpobj.get_bstar(q=q,omega=1.0) for q in range (9,13)]
    # betas=np.linspace(betas[0],betas[-1],10)
    for beta in betas:
        print("beta = {:.3f} ".format(beta))
        #add in marginals aligned with ground truth
        bpobj.run_modbp(beta=beta, q=qmax, niter=0, resgamma=1.0, omega=1.0,reset=True)
        start_vec=get_starting_partition(collapse_graph,gamma=1.0)
        # start_vec=collapse_graph.merged_comm_vec
        print('AMI start_vec', collapse_graph.get_AMI_with_communities(start_vec))
        ground_margs=create_marginals_from_comvec(start_vec,SNR=10,
                                                    q=qmax)
        new_belief = bpobj._create_beliefs_from_marginals(ground_margs)
        bpobj._set_beliefs(new_belief)
        bpobj.run_modbp(beta=beta, niter=600, q=qmax,reset=False,
                        resgamma=1.0, omega=1.0, anneal_omega=True)
        rm_df = bpobj.retrieval_modularities
        print(rm_df.loc[rm_df.shape[0]-1,['beta','niters','is_trivial','avg_entropy','AMI',"AMI_layer_avg",'bethe_free_energy']])

    rm_df = bpobj.retrieval_modularities
    rm_df.to_csv("beta_scan_q2.csv")
    print(rm_df.loc[:,['beta','niters','is_trivial','avg_entropy','AMI',"AMI_layer_avg",'bethe_free_energy']])


    idx=rm_df['bethe_free_energy'].idxmax()
    new_margs = expand_marginals(collapse_graph, bpobj.marginals[idx])
    #run on original graph
    bpobj2 = modbp.ModularityBP(mlgraph=multipex, use_effective=False, align_communities_across_layers_multiplex=False,align_communities_across_layers_temporal=False)

    beta2 = bpobj2.get_bstar(q=ncoms, omega=1.0)
    print("beta = {:.3f} ".format(beta2))

    bpobj2.run_modbp(beta=beta2, q=qmax, niter=0, resgamma=1.0, omega=1.0)
    new_belief = bpobj2._create_beliefs_from_marginals(new_margs)
    bpobj2._set_beliefs(new_belief)
    bpobj2.run_modbp(beta=beta2, q=qmax, niter=200, resgamma=1.0, omega=1.0, reset=False)
    rm_df = bpobj2.retrieval_modularities
    print(rm_df.loc[rm_df.shape[0]-1,['beta','niters','is_trivial','avg_entropy','AMI',"AMI_layer_avg"]])


    S=call_gen_louvain(collapse_graph,gamma=1.0,omega=3)
    print(S)
    print("AMI_layer_matlab = {:.3f} , AMI = {:.3f} ".format(collapse_graph.get_AMI_layer_avg_with_communities(S),
                                                             collapse_graph.get_AMI_with_communities(S)))

    return 0

def test_alternating_bpruns():
    n = 200
    nlayers = 15
    mu = .8
    p_eta = 1.0
    ncoms = 2
    omega = 1.0

    t = time()

    load = False
    if not load:
        # multipex = create_multiplex_graph_matlab(n_nodes=n, mu=mu, p_in=p_eta,nblocks=3,
        #                                          nlayers=nlayers, ncoms=ncoms)
        multipex = create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                          n_layers=nlayers, ncoms=ncoms)
        with gzip.open("working_graph.gz", 'wb') as fh:
            pickle.dump(multipex, fh)
    else:
        with gzip.open("working_graph.gz", 'rb') as fh:
            multipex = pickle.load(fh)

    multipex.reorder_nodes()
    print(np.unique(multipex.comm_vec, return_counts=True))

    rand_coms = np.array([range(n) for _ in range(nlayers)]).flatten()
    # rand_coms = np.array(range(n*nlayers))

    collapse_graph = modbp.convertMultilayertoMergedMultilayer(multipex)
    collapse_graph = collapse_graph.createCollapsedGraph(rand_coms, maintain_sparsity=False)

    ig_col = collapse_graph._export_to_igraph()
    ig_col.save("working_graph.graphml.gz")
    # collapse_graph.normalize_edge_weights(omega=1.0)

    print('time creating graph: {:.3f}'.format(time() - t))

    qmax = ncoms + 2

    bpobj = modbp.ModularityBP(mlgraph=collapse_graph, use_effective=False,
                               align_communities_across_layers_temporal=False,
                               align_communities_across_layers_multiplex=False)
    bpobj2 = modbp.ModularityBP(mlgraph=multipex, use_effective=False,
                               align_communities_across_layers_temporal=False,
                               align_communities_across_layers_multiplex=False)


    for q in range(4,5):
        beta1=bpobj.get_bstar(q=q,omega=1.0)
        beta2=bpobj.get_bstar(q=q,omega=1.0)
        start_vec=get_starting_partition(collapse_graph,gamma=1.0)
        # start_vec=collapse_graph.merged_comm_vec
        print('AMI start_vec', collapse_graph.get_AMI_with_communities(start_vec))
        ground_margs=create_marginals_from_comvec(start_vec,SNR=10,
                                                    q=qmax)
        new_belief = bpobj._create_beliefs_from_marginals(ground_margs)
        bpobj._set_beliefs(new_belief)
        bpobj.run_modbp(beta=beta1, niter=600, q=qmax,reset=False,
                        resgamma=1.0, omega=1.0, anneal_omega=True)





if __name__=="__main__":
    # test_run_modbp_on_collapse()
    test_on_multiplex_block()