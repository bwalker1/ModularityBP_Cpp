import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS

import sklearn.metrics as skm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import modbp
import seaborn as sbn
import logging
import gzip,pickle
import os,sys,re
from time import time
import itertools
sys.path.append(os.path.abspath("../multilayer_benchmark_matlab"))
from create_multiplex_functions import create_multiplex_graph
from create_multiplex_functions import get_non_backtracking_nodes
from create_multiplex_functions import create_multiplex_graph_matlab
from create_multiplex_functions import call_gen_louvain
from create_multiplex_functions import run_ZMBP_on_graph
from create_multiplex_functions import get_starting_partition_modularity
from create_multiplex_functions import get_starting_partition_multimodbp

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







def test_on_multiplex_block():
    n = 200
    nlayers = 20
    mu = .85
    p_eta = 1.0
    ncoms = 2
    omega = 1.0

    t = time()

    load = True
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

    bstars=[ bpobj.get_bstar(q=q,omega=1.0) for q in range (2,6)]
    betas=np.linspace(bstars[0],bstars[-1],10)
    for beta in betas:
        print("beta = {:.3f} ".format(beta))
        #add in marginals aligned with ground truth
        # start_vec=get_starting_partition(collapse_graph,gamma=1.0)
        start_vec=collapse_graph.merged_comm_vec
        print('AMI start_vec', collapse_graph.get_AMI_with_communities(start_vec))
        ground_margs=create_marginals_from_comvec(start_vec,SNR=100,
                                                    q=qmax)

        bpobj.run_modbp(beta=beta, niter=400, q=qmax,reset=False,
                        # starting_marginals=ground_margs,
                        dumping_rate=.1,
                        resgamma=1.0, omega=1.0, anneal_omega=True)
        rm_df = bpobj.retrieval_modularities
        print(rm_df.loc[rm_df.shape[0]-1,['beta','niters','converged','is_trivial','avg_entropy','AMI',"AMI_layer_avg",'bethe_free_energy']])

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

def collapse_over_interedges_same_community(graph,partition):
    N=len(partition)
    #each node starts in it's own community
    outgroups=dict(zip(range(N),[frozenset([i]) for i in range(N)]))
    for e in graph.interlayer_edges:
        if partition[e[0]] == partition[e[1]]:
            outgroups[e[0]] = outgroups[e[0]] | set([e[1]])
            outgroups[e[1]] = outgroups[e[1]] | set([e[0]])

    final_groups=list(set(outgroups.values()))
    group2ind=dict(zip(final_groups,range(len(final_groups))))

    collapse_vec=list(map(lambda x: group2ind[outgroups[x]],range(N)))
    return collapse_vec

def test_alternating_bpruns():
    n = 1000
    nlayers = 15
    mu = .7
    p_eta = 1.0
    ncoms = 10
    omega = 4

    t = time()

    load = False
    if not load:
        # multiplex = create_multiplex_graph_matlab(n_nodes=n, mu=mu, p_in=p_eta,nblocks=3,
        #                                          nlayers=nlayers, ncoms=ncoms)
        multiplex = create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                          n_layers=nlayers, ncoms=ncoms)
        with gzip.open("working_graph.gz", 'wb') as fh:
            pickle.dump(multiplex, fh)
    else:
        with gzip.open("working_graph.gz", 'rb') as fh:
            multiplex = pickle.load(fh)

    multiplex.reorder_nodes()

    print('interlayer avg: {:.3f}'.format(np.mean(multiplex.interdegrees)))
    print('intralayer avg: {:.3f}'.format(np.mean(multiplex.intradegrees)))
    print('total_edge_weight:{:.3f}'.format(multiplex.totaledgeweight))

    plt.close()
    f,a=plt.subplots(1,1,figsize=(6,6))
    degs,cnts=np.unique(multiplex.intradegrees,return_counts=True)
    plt.bar(x=degs,height=cnts)
    plt.show()
    print(np.unique(multiplex.comm_vec, return_counts=True))
    print('time creating graph: {:.3f}'.format(time() - t))

    qmax = 12

    bpobj = modbp.ModularityBP(mlgraph=multiplex, use_effective=True,
                               align_communities_across_layers_temporal=False,
                               align_communities_across_layers_multiplex=True)

    t=time()
    start_vec = get_starting_partition_modularity(multiplex, gamma=1.0,omega=1.0,q=ncoms)
    print('time creating starting vec:{:.3f}'.format(time()-t))
    print('AMI start_vec', multiplex.get_AMI_with_communities(start_vec))
    ground_margs = create_marginals_from_comvec(start_vec, SNR=5,
                                                q=qmax)
    bstars=[bpobj.get_bstar(q=q, omega=1.0) for q in range(1, 14)]
    print('bstars',bstars)
    not_converged=0
    #
    # for beta in np.linspace(bstars[0],bstars[-1],10):
    # for beta in np.linspace(.05, 2, 20):
    for beta in bstars:
        bpobj.run_modbp(beta=beta, niter=300, q=qmax, reset=True,
                        starting_marginals=ground_margs,
                        dumping_rate=1.0,
                        resgamma=1.0, omega=1.0)

        rm_df = bpobj.retrieval_modularities
        print(rm_df.loc[rm_df.shape[0] - 1, ['beta', 'niters','is_trivial', 'avg_entropy', 'AMI', "AMI_layer_avg",'converged']])
        if rm_df.loc[rm_df.shape[0]-1,'converged']==False:
            not_converged+=1


    plt.close()
    f,a=plt.subplots(1,1,figsize=(4,4))
    a.plot(rm_df['beta'],rm_df['niters'])
    a2=a.twinx()
    a2.scatter(rm_df['beta'],rm_df["AMI"],color='r')
    # a2.scatter(rm_df['beta'],rm_df["bethe_free_energy"],color='purple',marker='x')

    a2.vlines(x=bstars,ymin=0,ymax=1,linestyle='--')
    plt.show()

    S = call_gen_louvain(multiplex, gamma=1.0, omega=3)
    print(S)
    print("AMI_layer_matlab = {:.3f} , AMI = {:.3f} ".format(multiplex.get_AMI_layer_avg_with_communities(S),
                                                             multiplex.get_AMI_with_communities(S)))



def get_non_backtracking_modbp(mlgraph,q,beta,omega):

    nodes2edges = {}
    alloutgoingfactors = []
    edge2ind={}
    m=len(mlgraph.intralayer_edges)+len(mlgraph.interlayer_edges)

    for i,e in enumerate(itertools.chain(mlgraph.intralayer_edges,mlgraph.interlayer_edges)):
        if i<len(mlgraph.intralayer_edges):
            w=mlgraph.intralayer_weights[i]
        else:
            w=omega*mlgraph.interlayer_weights[i-len(mlgraph.intralayer_edges)]
        expfactor=np.exp(beta*w)
        alloutgoingfactors.append((expfactor-1)/(expfactor+q-1))
        # alloutgoingfactors.append(1)
        nodes2edges[e[0]]=nodes2edges.get(e[0],set([])) | set([e])
        nodes2edges[e[1]]=nodes2edges.get(e[1],set([])) | set([e])
        if e[0]<e[1]:
            edge2ind[e]=i
            edge2ind[(e[1],e[0])]=i+m
        else:
            edge2ind[e] = i + m
            edge2ind[(e[1], e[0])] = i


    node2incoming_inds={}
    node2outgoing_inds={}
    row_inds=[]
    col_inds=[]
    data=[]
    for i in range(mlgraph.N):
        node2incoming_inds[i] = node2incoming_inds.get(i, [])
        node2outgoing_inds[i] = node2outgoing_inds.get(i, [])

        try:
            cedges=nodes2edges[i]
        except KeyError:
            continue
        if len(cedges)==1:
            e = next(iter(cedges))
            en = 0 if e[0]==i else 1
            cind=edge2ind[(e[1-en],e[en])]
            cind_out=edge2ind[(e[en],e[1-en])]
            node2incoming_inds[i].append(cind)
            node2outgoing_inds[i].append(cind_out)

        for e1,e2 in itertools.combinations(cedges,2):

            e1w=alloutgoingfactors[edge2ind[e1]]
            e2w=alloutgoingfactors[edge2ind[e2]]
            #tell us which of the tuple represents current node
            e1n = 0 if e1[0]==i else 1
            e2n = 0 if e2[0]==i else 1

            #e1->e2->
            e1ind=edge2ind[(e1[1-e1n],e1[e1n])]
            e2ind=edge2ind[(e2[e2n],e2[1-e2n])]
            col_inds.append(e1ind)
            row_inds.append(e2ind)
            node2incoming_inds[i].append(e1ind)
            node2outgoing_inds[i].append(e2ind)
            data.append(e2w)

            #e2->e1->
            e1ind = edge2ind[(e1[e1n], e1[1-e1n])]
            e2ind = edge2ind[(e2[1-e2n], e2[e2n])]
            col_inds.append(e2ind)
            row_inds.append(e1ind)
            node2incoming_inds[i].append(e2ind)
            node2outgoing_inds[i].append(e1ind)

            data.append(e1w)

    for i,vals in node2incoming_inds.items():
        node2incoming_inds[i]=list(set(vals))

    for i, vals in node2outgoing_inds.items():
        node2outgoing_inds[i] = list(set(vals))

    nonBacktrack=sparse.csr_matrix((data,(row_inds,col_inds)),shape=(2*m,2*m),dtype=float)

    return nonBacktrack,node2incoming_inds,node2outgoing_inds




def test_non_backtracking_cluster():
    n = 200
    nlayers = 1
    mu = .1
    p_eta = 1.0
    ncoms = 3
    omega = 1.0
    gamma = 1.0

    t = time()

    load = False
    if not load:
        multiplex = modbp.generate_planted_partitions_dynamic_sbm(n=n,epsilon=mu,c=6,ncoms=ncoms,
                                                                  nlayers=nlayers,eta=p_eta)

        # multiplex = create_multiplex_graph_matlab(n_nodes=n, mu=mu, p_in=p_eta,nblocks=3,
        #                                          nlayers=nlayers, ncoms=ncoms)
        # multiplex = create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
        #                                    n_layers=nlayers, ncoms=ncoms)
        with gzip.open("working_graph.gz", 'wb') as fh:
            pickle.dump(multiplex, fh)
    else:
        with gzip.open("working_graph.gz", 'rb') as fh:
            multiplex = pickle.load(fh)

    # multiplex.reorder_nodes()
    print(np.unique(multiplex.comm_vec,return_counts=True))
    t=time()
    nbtrack=get_non_backtracking(multiplex)
    print('nbtrack',nbtrack.shape)
    print('nbtrack non-zero',nbtrack.nnz)

    print('time calculating B: {:.3f}'.format(time()-t))
    t=time()


    vals,vecs=slinagl.eigs(nbtrack,k=4,which='LR')
    # negvals,negvecs=slinagl.eigs(nbtrack,k=4,which='SR')
    # print(negvals)
    # vals=np.append(np.real(vals),np.abs(negvals))
    # vecs=np.hstack([vecs,negvecs])
    # vals=np.abs(np.real(vals))

    inds=list(range(0,vecs.shape[0],2))
    inds=list(range(n,vecs.shape[0]))

    vecs=vecs[inds,:]

    vecs=np.array(vecs)
    vecs[:,np.flip(np.argsort(vals))]
    vals=np.flip(np.sort(vals))
    print(vals)

    print('time calculating B eigen: {:.3f}'.format(time()-t))
    coms=np.unique(multiplex.comm_vec)
    colors=sbn.color_palette("Set1",n_colors=len(coms))
    com2col=dict(zip(coms,colors))
    color_vec=list(map(lambda x:com2col[x],multiplex.comm_vec))
    vec2plot=np.real(vecs)

    cmap=sbn.cubehelix_palette(as_cmap=True,light=1.0)
    # plt.close()
    # f,a=plt.subplots(1,2,figsize=(6,3))
    # a=plt.subplot(1,2,1)
    # a.scatter(vecs[:,0],vecs[:,1],color=color_vec)
    #
    # # a=plt.subplot(1,2,2)
    # a = f.add_subplot(122, projection='3d')
    #
    # a.scatter(vec2plot[:,0].flatten(),vec2plot[:,1].flatten(),vec2plot[:,2].flatten(),color=color_vec)
    # plt.show()


    kmeans = KMeans(n_clusters=ncoms, random_state=0).fit(np.real(vecs[:,range(0,ncoms)]))
    print("Non-backtrack AMI",skm.adjusted_mutual_info_score(multiplex.comm_vec,kmeans.labels_))
    S=get_starting_partition(multiplex,gamma=gamma,omega=omega,q=ncoms)
    print("Mod Matrix AMI:",skm.adjusted_mutual_info_score(multiplex.comm_vec,S))

    return

def test_non_backtracking_edges_cluster():
    n = 200
    nlayers = 10
    mu = .1
    p_eta = .5
    ncoms = 3
    omega = 5
    gamma = 1.5

    t = time()

    load = False
    if not load:
        # multiplex = modbp.generate_planted_partitions_dynamic_sbm(n=n,epsilon=mu,c=6,ncoms=ncoms,
        #                                                           nlayers=nlayers,eta=p_eta)

        # multiplex = create_multiplex_graph_matlab(n_nodes=n, mu=mu, p_in=p_eta,nblocks=3,
        #                                          nlayers=nlayers, ncoms=ncoms)
        multiplex = create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                           n_layers=nlayers, ncoms=ncoms)
        with gzip.open("working_graph.gz", 'wb') as fh:
            pickle.dump(multiplex, fh)
    else:
        with gzip.open("working_graph.gz", 'rb') as fh:
            multiplex = pickle.load(fh)

    # multiplex.reorder_nodes()
    print(np.unique(multiplex.comm_vec,return_counts=True))
    t=time()
    beta=.05
    t = time()
    # for beta in np.linspace(.01,1,10):
    print(beta)
    nbtrack,node_in_inds,node_out_inds=get_non_backtracking_modbp(multiplex,q=ncoms,beta=beta,omega=omega)
    vals,vecs=slinalg.eigs(nbtrack,k=ncoms,which='LR')
    vecs=vecs[:,np.flip(np.argsort(np.real(vals)))]
    comb_vecs=np.zeros((multiplex.N,vecs.shape[1]))
    # nbtrack_comb=np.zeros((2*multiplex.N,2*multiplex.N))
    for i in range(multiplex.N):
        in_inds=node_in_inds[i]
        if len(in_inds)!=0:
            comb_vecs[i,:]=np.sum(vecs[in_inds,:],axis=0)
    # vec2plot=np.real(comb_vecs)
    print('time calculating B edges + eigen: {:.3f}'.format(time()-t))
    # kmeans = KMeans(n_clusters=ncoms).fit(vec2plot[:, range(0, ncoms)])
    # print("Edges Non-backtrack AMI", skm.adjusted_mutual_info_score(multiplex.comm_vec, kmeans.labels_))

    t=time()
    nbtrack2=get_non_backtracking_nodes(multiplex,gamma=gamma,omega=omega)
    vals2,vecs2=slinalg.eigs(nbtrack2,k=ncoms,which='LR')
    inds=list(range(multiplex.N,vecs2.shape[0]))
    vecs2=vecs2[inds,:]
    vecs2=vecs2[:,np.flip(np.argsort(np.real(vals2)))]
    print('time calculating B nodes + eigen: {:.3f}'.format(time()-t))

    print('edges', vals)
    print('nodes',vals2)

    #modularity matrix
    A, C = multiplex.to_scipy_csr()
    A += A.T
    C += C.T
    P = multiplex.create_null_adj()
    B = A - gamma * P + omega * C
    evals, evecs = slinalg.eigs(B, k=ncoms, which='LR')
    evecs = np.array(evecs)
    evecs2plot = np.real(evecs[:, np.flip(np.argsort(evals))])



    coms=np.unique(multiplex.comm_vec)
    colors=sbn.color_palette("Set1",n_colors=len(coms))
    com2col=dict(zip(coms,colors))
    color_vec=list(map(lambda x:com2col[x],multiplex.comm_vec))

    vec2plot=np.real(comb_vecs)
    vec2plot2=np.real(vecs2)

    print(vec2plot.shape)
    print(vec2plot2.shape)


    kmeans = KMeans(n_clusters=ncoms).fit(vec2plot[:,range(0,ncoms)])
    print("Edges Non-backtrack KMeans AMI",skm.adjusted_mutual_info_score(multiplex.comm_vec,kmeans.labels_))
    spectral = SpectralClustering(n_clusters=ncoms,affinity='rbf').fit(vec2plot[:, range(0, ncoms)])
    print("Edges Non-backtrack SC AMI", skm.adjusted_mutual_info_score(multiplex.comm_vec, spectral.labels_))

    kmeans = KMeans(n_clusters=ncoms).fit(vec2plot2[:, range(0, ncoms)])
    print("Nodes Non-backtrack KMeans AMI", skm.adjusted_mutual_info_score(multiplex.comm_vec, kmeans.labels_))

    meanshift = MeanShift(bin_seeding=True).fit(vec2plot2[:, range(0, ncoms)])
    print("Nodes Non-backtrack MeanShift AMI", skm.adjusted_mutual_info_score(multiplex.comm_vec, meanshift.labels_))


    optics = OPTICS().fit(vec2plot2[:, range(0, ncoms)])
    print("Nodes Non-backtrack OPTICS AMI", skm.adjusted_mutual_info_score(multiplex.comm_vec, optics.labels_))
    #
    # spect_clust=SpectralClustering(n_clusters=ncoms,affinity='rbf',n_neighbors=10).fit(vec2plot2[:, range(0, ncoms)])
    # print("Nodes Non-backtrack SC AMI", skm.adjusted_mutual_info_score(multiplex.comm_vec, spect_clust.labels_))

    gauss_mix_labels=GaussianMixture(n_components=ncoms).fit_predict(vec2plot2[:, range(0, ncoms)])
    print("Nodes Non-backtrack GaussMix AMI", skm.adjusted_mutual_info_score(multiplex.comm_vec, gauss_mix_labels))

    kmeans = KMeans(n_clusters=ncoms).fit(evecs2plot)
    print("Mod Matrix Kmeans AMI:",skm.adjusted_mutual_info_score(multiplex.comm_vec,kmeans.labels_))
    meanshift = MeanShift(bin_seeding=True).fit(evecs2plot)
    print("Mod Matrix MeanShift AMI:", skm.adjusted_mutual_info_score(multiplex.comm_vec, meanshift.labels_))

    print('edges',vec2plot[:2,:])
    print('nodes',vec2plot2[:2,:])


    cmap=sbn.cubehelix_palette(as_cmap=True,light=1.0)
    plt.close()
    f,a=plt.subplots(1,3,figsize=(18,3))
    # a=plt.subplot(1,2,1)
    # a.scatter(vec2plot[:,0].flatten(),vec2plot[:,1].flatten(),color=color_vec)

    a = f.add_subplot(131, projection='3d')
    a.set_title("Edges B")
    a.scatter(vec2plot[:, 0].flatten(), vec2plot[:, 1].flatten(), vec2plot[:, 2].flatten(), color=color_vec)

    # a=plt.subplot(1,2,2)
    a = f.add_subplot(132, projection='3d')
    a.set_title("Nodes B")
    a.scatter(vec2plot2[:,0].flatten(),vec2plot2[:,1].flatten(),vec2plot2[:,2].flatten(),color=color_vec)

    a = f.add_subplot(133, projection='3d')
    a.set_title("modularity B")
    a.scatter(evecs2plot[:, 0].flatten(), evecs2plot[:, 1].flatten(), evecs2plot[:, 2].flatten(), color=color_vec)
    plt.show()



    return

def test_ZM_on_collapsed():
    n = 200
    nlayers = 15
    mu = .85
    p_eta = 1.0
    ncoms = 2
    omega = 1.0

    t = time()

    load = False
    if not load:
        # multiplex = create_multiplex_graph_matlab(n_nodes=n, mu=mu, p_in=p_eta,nblocks=3,
        #                                          nlayers=nlayers, ncoms=ncoms)
        multiplex = create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                           n_layers=nlayers, ncoms=ncoms)
        with gzip.open("working_graph.gz", 'wb') as fh:
            pickle.dump(multiplex, fh)
    else:
        with gzip.open("working_graph.gz", 'rb') as fh:
            multiplex = pickle.load(fh)

    multiplex.reorder_nodes()
    print(np.unique(multiplex.comm_vec, return_counts=True))
    rand_coms = np.array([range(n) for _ in range(nlayers)]).flatten()
    collapse_graph = modbp.convertMultilayertoMergedMultilayer(multiplex)
    collapse_graph = collapse_graph.createCollapsedGraph(rand_coms, maintain_sparsity=False)
    ig_col = collapse_graph._export_to_igraph()
    bpobj = modbp.ModularityBP(mlgraph=collapse_graph, use_effective=False,
                               align_communities_across_layers_temporal=False,
                               align_communities_across_layers_multiplex=False)
    bstars=[bpobj.get_bstar(q=q,omega=1.0) for q in range(2,8)]
    for beta in np.linspace(bstars[0]-.1, bstars[-1], 10):
        print('beta', beta)
        t=time()
        niters,cmarginals=run_ZMBP_on_graph(ig_col,q=ncoms,beta=beta,niters=2000)
        cpart=np.argmax(cmarginals,axis=1)
        print("ZM modbp AMI: {:.3f}".format(collapse_graph.get_AMI_layer_avg_with_communities(cpart)))
        t2=time()-t
        print("time to run {:d} iters: {:.3f}.  iters/s = {:.3f}".format(niters,t2,niters/t2))
        t=time()
        bpobj.run_modbp(beta=beta, niter=1000, q=ncoms,
                        # starting_marginals=cmarginals,
                        resgamma=.5, omega=1.0)
        t2=time()-t
        rm_df=bpobj.retrieval_modularities
        niters = bpobj.retrieval_modularities.loc[rm_df.shape[0] - 1,'niters']
        print("time to run {:.1f} iters: {:.3f}.  iters/s = {:.3f}".format(niters,t2,niters/t2))

        # rm_df.loc[rm_df.shape[0] - 1, 'niters'] = niters

        print(rm_df.loc[
            rm_df.shape[0] - 1, ['beta', 'niters', 'is_trivial', 'avg_entropy', 'AMI', "AMI_layer_avg", 'converged']])

    S = call_gen_louvain(multiplex, gamma=1.0, omega=3)
    print(S)
    print("AMI_layer_matlab = {:.3f} , AMI = {:.3f} ".format(multiplex.get_AMI_layer_avg_with_communities(S),
                                                             multiplex.get_AMI_with_communities(S)))

if __name__=="__main__":
    # test_run_modbp_on_collapse()
    # test_on_multiplex_block()
    test_alternating_bpruns()
    # test_ZM_on_collapsed()
    # test_non_backtracking_cluster()
    # test_non_backtracking_edges_cluster()