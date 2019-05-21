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

# clusterdir = "/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/"
clusterdir = "/home/wweir/Modularity_BP_proj/ModularityBP_Cpp" #lccc
# clusterdir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/" #for testing locally

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



    finoutdir = os.path.join(clusterdir, 'test/modbpdata/SBM_test_data_n{:}_q{:d}_nt{:}'.format(n, q, ntrials))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)
    outfile = os.path.join(finoutdir,
                           "sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.2f}_gamma{:.2f}.csv".format(n, q, nlayers,
                                                                                                         eta, ep, omega,
                                                                                                         gamma))
    print(outfile)
    qmax = 2 * q
    for trial in range(ntrials):
        mgraph=modbp.generate_planted_partitions_dynamic_sbm(n,ncoms=q,epsilon=ep,
                                                             eta=eta,nlayers=nlayers)
        mlbp = modbp.ModularityBP(mlgraph=mgraph, use_effective=True, accuracy_off=False)

        bstars = [mlbp.get_bstar(q_i, omega) for q_i in range(2, qmax+1)]

        betas=bstars
        for j,beta in enumerate(betas):
            mlbp.run_modbp(beta=beta, niter=2000, q=qmax, resgamma=gamma, omega=omega,reset=True)
            mlbp_rm = mlbp.retrieval_modularities

            mlbp_rm['trial']=trial
            mlbp_rm['ep'] = ep
            mlbp_rm['eta'] = eta
            mlbp_rm['n'] = n
            mlbp_rm['q_true'] = q
            #append as we complete beta of each trial
            if trial == 0 and j==0:
                with open(outfile, 'w') as fh:
                    mlbp_rm.to_csv(fh, header=True)
            else:
                with open(outfile, 'a') as fh:  # writeout as we go
                    mlbp_rm.iloc[[-1], :].to_csv(fh, header=False)


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
