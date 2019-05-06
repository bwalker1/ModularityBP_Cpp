from __future__ import division
# from context import modbp
import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import sys
from subprocess import Popen, PIPE
import re
import os
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import traceback

clusterdir = "/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/"
#clusterdir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/" #for testing locally

# python run_sbm_ml_test.py 100 2 10 .1 5 .1 1 0.5 1.0
# python run_sbm_ml_test.py 250 2 20 0 10 0.2 1 2.0 .5
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
                                   'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial','converged'])


    finoutdir = os.path.join(clusterdir, 'test/modbpdata/SBM_test_data_n{:}_q{:d}_nt{:}'.format(n, q, ntrials))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)
    outfile = os.path.join(finoutdir,
                           "sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.2f}_gamma{:.2f}.csv".format(n, q, nlayers,
                                                                                                         eta, ep, omega,
                                                                                                         gamma))

    qmax = 2 * q
    #qmax = q
    for trial in range(ntrials):
        ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
        mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.layer_vec, ml_sbm.interedges,
                                       comm_vec=ml_sbm.get_all_layers_block())

        mlbp = modbp.ModularityBP(mlgraph=mgraph, use_effective=True, accuracy_off=False)

        # mlbp.run_modbp(beta=beta, niter=1000, q=qmax, resgamma=gamma, omega=omega)
        bstars = [mlbp.get_bstar(q_i, omega) for q_i in range(2, qmax+1)]
        # betas = np.linspace(bstars[0], bstars[-1], 3*(qmax-2))
        #betas=[mlbp.get_bstar(q,omega)]
        betas=bstars
        for beta in betas:
            mlbp.run_modbp(beta=beta, niter=2000, q=qmax, resgamma=gamma, omega=omega,reset=True)
            mlbp_rm = mlbp.retrieval_modularities
            # print(mlbp_rm.loc[mlbp_rm.shape[0]-1,'AMI'])
            # plt.close()
            # f,a=plt.subplots(1,2)
            # mlbp.plot_communities(ax=a[0])
            # mlbp.plot_communities(ind=mlbp_rm.shape[0]-1,ax=a[1])
            # plt.show()

        if trial==0:
            with open(outfile,'w') as fh:
                mlbp_rm.to_csv(fh,header=True)
        else:
            with open(outfile,'a') as fh: #writeout as we go
                mlbp_rm.to_csv(fh, header=False)



        # these are the non-trivial ones
        # minidx = mlbp_rm[mlbp_rm['converged'] == True][
        #     'retrieval_modularity']  # & ~mlbp_rm['is_trivial'] ]['retrieval_modularity']
        # cind = output.shape[0]

        # if minidx.shape[0] == 0:
        #     output.loc[cind, ['ep', 'eta', 'resgamma', 'omega','converged']] = [ep, eta, gamma, omega,False]
        #     output.loc[cind, ['niters']] = 1000
        #     continue
        # minidx = minidx.idxmax()

        # output.loc[cind, ['beta', 'resgamma', 'omega', 'niters', 'AMI', 'AMI_layer_avg', 'retrieval_modularity',
        #                   'bethe_free_energy', 'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial','converged']] = \
        #     mlbp_rm.loc[
        #         minidx, ['beta', 'resgamma', 'omega', 'niters', 'AMI', 'AMI_layer_avg', 'retrieval_modularity',
        #                  'bethe_free_energy', 'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial','converged']]
        # output.loc[cind, ['ep', 'eta']] = [ep, eta]
        # if trial==0:
        #     with open(outfile,'w') as fh:
        #         output.to_csv(fh,header=True)
        # else:
        #     with open(outfile,'a') as fh: #writeout as we go
        #         output.iloc[[-1], :].to_csv(fh, header=False)


    return 0



def pydebug(type, value, tb):
    n = int(sys.argv[1])
    q = int(sys.argv[2])
    nlayers = int(sys.argv[3])
    eta = float(sys.argv[4])
    c = float(sys.argv[5])
    ep = float(sys.argv[6])
    ntrials = int(sys.argv[7])
    omega = float(sys.argv[8])
    gamma = float(sys.argv[9])


    finoutdir = os.path.join(clusterdir, 'test/modbpdata/SBM_test_data_n{:}_q{:d}_nt{:}/errors'.format(n, q, ntrials))

    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)
    erroroutfile = os.path.join(finoutdir,
                            "sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.2f}_gamma{:.2f}.error".format(n, q, nlayers,
                                                                                                          eta, ep,
                                                                                                          omega, gamma))
    with open(erroroutfile, 'w') as fh:
        #write error out to file
        # fh.write("Error type:" + str(type) + ": " + str(value))
        traceback.print_exception(type,value,tb,file=fh)
        # traceback.print_tb(tb,file=fh)

if __name__=='__main__':
        # sys.excepthook=pydebug
        sys.exit(main())
