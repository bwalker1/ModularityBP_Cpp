from __future__ import division
from context import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import sys
from subprocess import Popen, PIPE
import re
import os
import sklearn.metrics as skm
import matplotlib.pyplot as plt


# python run_sbm_ml_test.py 100 2 10 .1 5 .1 2 0.5 1.0

def main():
    # generate a graph and then run it some number of times
    n = int(sys.argv[1])
    q = int(sys.argv[2])
    nlayers = int(sys.argv[3])
    eta = float(sys.argv[4])
    c = float(sys.argv[5])
    ep = float(sys.argv[6])
    ntrials = int(sys.argv[7])
    omega = float(sys.argv[8])
    gamma = float(sys.argv[9])

    nblocks = q

    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    output = pd.DataFrame(columns=['ep', 'eta', 'beta', 'resgamma', 'omega', 'niters',
                                   'AMI', 'AMI_layer_avg', 'retrieval_modularity', 'bethe_free_energy',
                                   'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial'])

    clusterdir = "/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/"
    # clusterdir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/" #for testing locally
    finoutdir = os.path.join(clusterdir, 'test/modbpdata/SBM_test_data_n{:}_q{:d}_nt{:}'.format(n, q, ntrials))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)
    outfile = os.path.join(finoutdir,
                           "sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.2f}_gamma{:.2f}.csv".format(n, q, nlayers,
                                                                                                         eta, ep, omega,
                                                                                                         gamma))
    qmax = 2 * q
    for trial in range(ntrials):
        ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
        mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.layer_vec, ml_sbm.interedges,
                                       comm_vec=ml_sbm.get_all_layers_block())

        mlbp = modbp.ModularityBP(mlgraph=mgraph, use_effective=True, accuracy_off=False)

        # mlbp.run_modbp(beta=beta, niter=1000, q=qmax, resgamma=gamma, omega=omega)
        bstar = mlbp.get_bstar(q, omega)

        bstars = [mlbp.get_bstar(q, omega) for q in range(2, qmax + 1)]
        betas = np.linspace(bstars[0], bstars[-1], 3*len(bstars))
        for beta in betas:
            mlbp.run_modbp(beta=bstar, niter=1000, q=qmax, resgamma=gamma, omega=omega)
            mlbp_rm = mlbp.retrieval_modularities

    # these are the non-trivial ones
        minidx = mlbp_rm[mlbp_rm['niters'] < 1000]['retrieval_modularity']  # & ~mlbp_rm['is_trivial'] ]['retrieval_modularity']
        cind = output.shape[0]

        if minidx.shape[0] == 0:
            output.loc[cind, ['ep', 'eta', 'resgamma', 'omega']] = [ep, eta, gamma, omega]
            output.loc[cind, ['niters']] = 1000
            continue
        minidx = minidx.idxmax()

        output.loc[cind, ['beta', 'resgamma', 'omega', 'niters', 'AMI', 'AMI_layer_avg', 'retrieval_modularity',
                      'bethe_free_energy', 'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial']] = \
        mlbp_rm.loc[
        minidx, ['beta', 'resgamma', 'omega', 'niters', 'AMI', 'AMI_layer_avg', 'retrieval_modularity',
                 'bethe_free_energy', 'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial']]
        output.loc[cind, ['ep', 'eta']] = [ep, eta]


    output.to_csv(outfile)
    return 0

if __name__ == '__main__':
    sys.exit(main())
