from __future__ import print_function
import scipy.io as scio
import scipy.sparse as sparse
import pandas as pd
import sys, os
import modbp
import numpy as np
import network_tools as nt
import matplotlib.pyplot as plt
import gzip
import pickle


def create_knn_from_adj(adj_mat, k, weight_func=None):
    # assume that final adj is undirected
    n = adj_mat.shape[0]
    #     out_adj = sparse.csr_matrix((n, n))
    out_adj = np.zeros((n, n))
    for i in range(n):
        closest_inds = np.argpartition(np.array(adj_mat[i, :].todense())[0], -k - 1)[-k - 1:]
        closest_inds = closest_inds[closest_inds != i]  # no self edges
        closest_inds = [x for x in closest_inds if adj_mat[i, x] > 0]
        # print(i,len(closest_inds))
        if weight_func is None:
            out_adj[i, closest_inds] = 1
            out_adj[closest_inds, i] = 1
        else:
            vals = np.array(list(map(lambda (x): weight_func(x), adj_mat[i, closest_inds].data)))
            for j,ind in enumerate(closest_inds):
                out_adj[i, ind] = vals[j]
                out_adj[ind, i] = vals[j]
    return out_adj

def adjacency_to_edges(A):
    nnz_inds = np.nonzero(A)
    nnzvals = np.array(A[nnz_inds])
    if len(nnzvals.shape) > 1:
        nnzvals = nnzvals[0]  # handle scipy sparse types
    return zip(nnz_inds[0], nnz_inds[1], nnzvals)

def main():
    gamma=float(sys.argv[1])
    omega=float(sys.argv[2])

    senate_dir = '/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/modularity_domains/multilayer_senate'
    #senate_dir = '/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/senate_data'

    senate_out_dir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/test/senate_data"
    #senate_out_dir='/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/senate_data'

    senate_data_file = os.path.join(senate_dir, 'multisenate0.5.mat')
    sendata = scio.loadmat(senate_data_file)


    A = sendata['A']
    num2keep = 50
    #num2keep = A.shape[0]
    A = A[:num2keep,:num2keep]
    C = sendata['C']
    C= C[:num2keep,:num2keep]
    sesid = sendata['Ssess'][:, 0]
    parties = sendata['Sparty'][:, 0]
    parties = parties[:num2keep]
    sessions = np.unique(sesid)
    sess2layer = dict(zip(sessions, range(len(sessions))))
    layer_vec = np.array(list(map(lambda x: sess2layer[x], sesid)))[:num2keep]

    k=6
    A_knn = create_knn_from_adj(A, k,weight_func=lambda (x): x)

    intra_edges = adjacency_to_edges(A_knn)
    inter_edges = adjacency_to_edges(C)


    # A_gtools=nt.create_gt_graph_from_adj(A_knn)
    # for e in inter_edges:
    #     cedge=A_gtools.add_edge(e[0],e[1])
    #     A_gtools.ep['weight'][cedge]=1.0/10
    #
    # A_gtools.save("senate_{}_knn.graphml.gz".format(k))

    mgraph = modbp.MultilayerGraph(interlayer_edges=inter_edges,
                                   intralayer_edges=intra_edges,
                                   layer_vec=layer_vec,directed=False)
    q_max_val = 20
    #gamma_vals = [.5, 1.0, 1.5 , 2 , 4]
    #omega_vals = [0.0, 1, 2, 4, 8]

    modbp_obj = modbp.ModularityBP(mlgraph=mgraph,use_effective=True,align_communities_across_layers=True,
                                   accuracy_off=True,comm_vec=parties)



    bstars = list(map(lambda(q): modbp_obj.get_bstar(q,omega=omega),range(4,q_max_val,4)))
    for beta in bstars:
        modbp_obj.run_modbp(beta=beta,q=q_max_val,niter=2000,
                            omega=omega,resgamma=gamma,reset=False)

    if not os.path.exists(senate_out_dir):
        os.makedirs(senate_out_dir)
    with gzip.open(os.path.join(senate_out_dir,"senate_partitions_{:.4f}_{:.4f}_.gz".format(gamma,omega)),'wb') as fh:
        pickle.dump(modbp_obj.partitions,fh)

    modrm=modbp_obj.retrieval_modularities
    modrm.to_csv("senate_ret_mod_df_{:.4f}_{:.4f}.csv".format(gamma,omega))

    return 0

if __name__=='__main__':
    sys.exit(main())