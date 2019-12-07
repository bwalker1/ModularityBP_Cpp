import pandas as pd
import modbp
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
from time import time
import gzip,pickle
import os,sys
sys.path.append(os.path.abspath("../multilayer_benchmark_matlab"))
from create_multiplex_functions import create_multiplex_graph
from create_multiplex_functions import create_multiplex_graph_matlab


def create_marginals_from_comvec(commvec,q=None,SNR=3):
    if q is None:
        q=len(np.unique(commvec))

    outmargs=np.zeros((len(commvec),q))
    for i in range(len(commvec)):
        currow=np.array([1 for _ in range(q)])
        currow[int(commvec[i])]=SNR
        currow=1/np.sum(currow)*currow
        outmargs[i,:]=currow
    return outmargs

def test_multiplex():
    n=1000
    nlayers=15
    mu=.9
    p_eta=1.0
    ncoms=10
    omega=1.0

    t = time()

    load = True
    if not load:
        multipex = create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                       n_layers=nlayers, ncoms=ncoms)
        with gzip.open("working_graph.gz", 'wb') as fh:
            pickle.dump(multipex, fh)
    else:
        with gzip.open("working_graph.gz", 'rb') as fh:
            multipex = pickle.load(fh)

    print('time creating graph: {:.3f}'.format(time() - t))


    multipex.reorder_nodes()

    bpobj=modbp.ModularityBP(multipex,comm_vec=multipex.comm_vec,
                             align_communities_across_layers_multiplex=False,
                             align_communities_across_layers_temporal=False,
                             use_effective=False)

    beta=bpobj.get_bstar(q=ncoms,omega=omega)
    bpobj.run_modbp(beta=beta,q=ncoms,niter=10,omega=omega)

    ground_margs=create_marginals_from_comvec(multipex.comm_vec,q=ncoms)
    print(bpobj.retrieval_modularities.head())
    newbeliefs=bpobj._create_beliefs_from_marginals(ground_margs)

    # cmargs=bpobj.marginals[0]

    # bpobj.partitions[0]=mixed_comvec

    plt.close()
    f, a = plt.subplots(1, 3, figsize=(8, 4))
    a = plt.subplot(1, 3, 1)
    multipex.plot_communities(ax=a)


    a = plt.subplot(1, 3, 2)
    a.set_title("before setting beliefs")
    bpobj.plot_communities(ind=0,ax=a)
    cami=skm.adjusted_mutual_info_score(bpobj.partitions[0],bpobj.graph.comm_vec)
    a.text(s='AMI={:.3f}'.format(cami), x=.1, y=.1, transform=a.transAxes)




    a = plt.subplot(1, 3, 3)
    a.set_title('After setting beliefs back')
    good_beliefs=np.array(bpobj._bpmod.getBeliefs())
    bpobj._set_beliefs(newbeliefs)
    bpobj.run_modbp(beta=beta,q=ncoms,niter=100,omega=omega,reset=False)
    bpobj.plot_communities(ind=1,ax=a)
    cami = skm.adjusted_mutual_info_score(bpobj.partitions[1], bpobj.graph.comm_vec)
    a.text(s='AMI={:.3f}'.format(cami), x=.1, y=.1, transform=a.transAxes)

    plt.show()
    # bpobj._switch_beliefs_bp(0)
    # # bpobj._bpmod.step()
    # cmargs = np.array(bpobj._bpmod.return_marginals())
    # bpobj.marginals[0] = cmargs
    # bpobj._get_community_distances(0)  # sets values in method
    # cpartition = bpobj._get_partition(0, True)
    # bpobj.partitions[0] = cpartition
    # bpobj.plot_communities(ind=0,ax=a)
    # cami=skm.adjusted_mutual_info_score(bpobj.partitions[0],bpobj.graph.comm_vec)
    # a.text(s='AMI={:.3f}'.format(cami), x=.1, y=.1, transform=a.transAxes)
    # plt.show()




    return 0

if __name__=='__main__':
    sys.exit(test_multiplex())