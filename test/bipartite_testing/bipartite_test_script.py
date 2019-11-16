import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import sys
import re
import os
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import igraph as ig



def main():
    n = 30
    m = 50
    p = .05
    bipart_g = ig.Graph.Random_Bipartite(n1=n, n2=m, p=p)

    adj = np.array(bipart_g.get_adjacency().data)
    # plt.close()
    # plt.pcolor(adj)
    # plt.show()

    bpclasses = [0] * n + [1] * m
    intra_edges = bipart_g.get_edgelist()
    mlgraph = modbp.MultilayerGraph(intralayer_edges=intra_edges,bipartite_classes=bpclasses,
                                    layer_vec=[0 for _ in range(n + m)])

    qmax=4
    bpobj=modbp.ModularityBP(mlgraph=mlgraph,
                             align_communities_across_layers_temporal=False)
    nruns=3
    betas=[ bpobj.get_bstar(q) for q in range(2,4) ]
    for beta in betas:
        for i in range(nruns):
            bpobj.run_modbp(beta,q=qmax,resgamma=1.0,niter=200,reset=True)


    rm_df = bpobj.retrieval_modularities

    return 0


if __name__=="__main__":

    sys.exit(main())