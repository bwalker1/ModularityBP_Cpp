import scipy.io as scio
import scipy.sparse as sparse
import pandas as pd
import sys,os
import subprocess
import re
import modbp
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import graph_tool as gt
import gzip
import pickle
import sklearn.metrics as skm
import graph_tool.draw as gtd
import champ
import matplotlib.colors as mc
import matplotlib.colorbar as mcb


senate_out_dir= '/experiments/senate_data'

part_dir='zippe_partitions_knn10'

def get_partition(ind, partitions_df_highcoms , download=False):
    omega = partitions_df_highcoms.loc[ind, 'omega']
    gamma = partitions_df_highcoms.loc[ind, 'resgamma']
    trial = partitions_df_highcoms.loc[ind][0]
    cfilename = 'senate_partitions_{:.4f}_{:.4f}_.gz'.format(gamma, omega)
    cfullpath = os.path.join(senate_out_dir, part_dir, cfilename)

    if download:
        lldir = "/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/experiments/senate_data/zippe_partitions_knn10"

        command = "rsync -varpzu wweir@longleaf.unc.edu:{} {}".format(os.path.join(lldir, cfilename),
                                                                      cfullpath)
        print(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        stdout, stderr = p.communicate()
        print(stdout)
        if stderr: print(stderr)

    # load

    print (cfullpath)
    with gzip.open(cfullpath, 'rb') as fh:
        cpart = pickle.load(fh)[trial]
    prtstr = 'part_{:.4f}_{:.4f}_{}'.format(gamma, omega, trial)
    return cpart, prtstr

def adjacency_to_edges(A):
    nnz_inds = np.nonzero(A)
    nnzvals = np.array(A[nnz_inds])
    if len(nnzvals.shape)>1:
        nnzvals=nnzvals[0] #handle scipy sparse types
    return zip(nnz_inds[0], nnz_inds[1], nnzvals)

def main():
    senate_out_dir = '/experiments/senate_data'
    partitions_df = pd.read_csv(os.path.join(senate_out_dir, 'merged_all_senate_ret_mod_dfs_knn10'))
    partitions_df_highcoms = partitions_df[partitions_df['num_coms'] > 2]
    partitions_df_highcoms.sort_values(by='bethe_free_energy', ascending=True, inplace=True)
    senate_dir = '/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/modularity_domains/multilayer_senate'
    senate_data_file = os.path.join(senate_dir, 'multisenate0.5.mat')
    sendata = scio.loadmat(senate_data_file)

    A = sendata['A']
    C = sendata['C']
    sesid = sendata['Ssess'][:, 0]
    parties = sendata['Sparty'][:, 0]
    SID = sendata['SID']
    sessions = np.unique(sesid)
    sess2layer = dict(zip(sessions, range(len(sessions))))
    layer_vec = np.array(list(map(lambda x: sess2layer[x], sesid)))

    # intralayer,interlayer=champ.create_multilayer_igraph_from_adjacency(A=A,C=C,layer_vec=layer_vec)

    k = 10
    intra_edge_file = os.path.join("senate_knn_{:d}_intra_edgelist.pickle".format(k))
    if os.path.exists(intra_edge_file):
        with gzip.open(intra_edge_file, 'rb') as fh:
            intra_edges = pickle.load(fh)
        inter_edges = adjacency_to_edges(C)

    else:
        raise AssertionError

    mgraph = modbp.MultilayerGraph(intra_edges, layer_vec, inter_edges, comm_vec=parties)

    A, C = mgraph.to_scipy_csr()
    P = np.zeros((mgraph.N, mgraph.N))
    cind = 0
    for layer in mgraph.layers:
        strength = np.array(layer.strength(weights='weight'))
        pcur = np.outer(strength, strength)
        pcur /= (2.0 * np.sum(layer.es['weight']))
        cinds = range(cind, cind + layer.vcount())
        P[np.ix_(cinds, cinds)] = pcur
        cind += layer.vcount()

    def get_champ_mod(A, P, C, partition, omega, gamma):
        ahat = champ.champ_functions.calculate_coefficient(adj_matrix=A, com_vec=partition)
        chat = champ.champ_functions.calculate_coefficient(adj_matrix=C, com_vec=partition)
        phat = champ.champ_functions.calculate_coefficient(adj_matrix=P, com_vec=partition)
        print("A:{:.2f},P:{:.2f},C:{:.2f}".format(ahat, phat, chat))
        factor = 2.0 * (np.sum(A) + np.sum(C))
        return (1.0 / factor) * (ahat - gamma * phat + omega * chat)

    ind2plot = partitions_df_highcoms.sort_values('AMI', ascending=False).iloc[:2, :].index
    for ind in ind2plot:
        cpart = get_partition(ind)[0]
        gamma = partitions_df_highcoms.loc[ind, 'resgamma']
        omega = partitions_df_highcoms.loc[ind, 'omega']

        mod1 = modbp.calc_modularity(graph=mgraph, omega=omega,
                                     partition=cpart, resgamma=gamma)

        champ_mod = get_champ_mod(A, P, C, cpart, omega, gamma)

        print(champ_mod, mod1, partitions_df_highcoms.loc[ind, 'retrieval_modularity'])
        print

    return 0


if __name__=='__main__':
    sys.exit(main())