import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import modbp
import seaborn as sbn
import logging
logging.basicConfig(level=logging.ERROR)

def test_collapse():
    n=100
    nlayers=5
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
    n = 100
    nlayers = 5
    ep = .1
    eta = 0
    c = 8
    ncoms = 3
    cmap = sbn.cubehelix_palette(as_cmap=True)
    graph = modbp.generate_planted_partitions_dynamic_sbm(n=n, epsilon=ep, eta=eta, c=c,
                                                          nlayers=nlayers, ncoms=ncoms)


    rand_coms=np.random.choice(range(ncoms),size=n)
    rand_coms=np.array([rand_coms for _ in range(nlayers)]).flatten()


    collapse_graph = modbp.convertMultilayertoMergedMultilayer(graph)
    collapse_graph = collapse_graph.createCollapsedGraph(rand_coms)

    bpobj=modbp.ModularityBP(mlgraph=collapse_graph,use_effective=False)
    beta=bpobj.get_bstar(q=3,omega=1.0)

    bpobj.run_modbp(beta=beta,niter=100,q=3,
                    resgamma=1.0,omega=1.0,anneal_omega=False)




    return 0



if __name__=="__main__":
    test_run_modbp_on_collapse()