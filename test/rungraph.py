from __future__ import division
from context import modbp
from time import time
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
import sys
from os.path import expanduser


if __name__ == "__main__":
    # generate a graph and then run it some number of times
    
    n = int(sys.argv[1])
    q = int(sys.argv[2])
    nlayers = int(sys.argv[3])
    eta = float(sys.argv[4])
    c = float(sys.argv[5])
    ep = float(sys.argv[6])
    ntrials = int(sys.argv[7])
    omega = float(sys.argv[8])
    
    nblocks = q
    
    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    


    accuracy = 0.0
    
    for trial in xrange(ntrials):
        ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
        mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.interedges, ml_sbm.layer_vec,comm_vec=ml_sbm.get_all_layers_block())
        mlbp = modbp.ModularityBP(mlgraph=mgraph)
        mlbp.run_modbp(q=q, beta=0, omega=omega, niter=500, reset=True)
        accuracy += mlbp.retrieval_modularities.loc[0,'Accuracy']
    
    accuracy /= ntrials
    
    f = open("{:s}/data/eps{:f}eta{:f}.dat".format(expanduser("~"),ep,eta),"wb")
    f.write("{:f} {:f} {:f}\n".format(ep,eta,accuracy))
    f.close()