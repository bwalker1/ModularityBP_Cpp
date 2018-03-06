from context import modbp
from time import time
import numpy as np
import igraph as ig

def main():
    n=200
    q=3
    nblocks=q
    pin=(5.0*q)/n
    pout=(.50*q)/n
    t=time()
    prob_mat=np.identity(nblocks)*pin + (np.ones((nblocks,nblocks))-np.identity(nblocks))*pout

    RSBM = modbp.RandomSBMGraph(n=n,comm_prob_mat=prob_mat)
    m= RSBM.m
    print("time to construct {:.4f}".format(time()-t))
    elist=RSBM.get_edgelist()
    elist.sort()
    pv=modbp.bp.PairVector(elist)
    bpgc=modbp.BP_Modularity(edgelist=pv, _n=n, q=q, beta=1, transform=False)
    old_marg=np.array(bpgc.return_marginals())
    for i in range(10):
        bpgc.step()
        new_marg=np.array(bpgc.return_marginals())
        print ("Change in margins {:d}: {:.3f}".format(i,np.sum(np.abs(old_marg-new_marg))/(1.0*q*n)))
        old_marg=new_marg

    color_dict={0:"red",1:"blue",2:'green'}
    RSBM.graph.vs['color']=map(lambda x : color_dict[np.argmax(x)],old_marg)
    ig.plot(RSBM.graph,layout=RSBM.graph.layout('kk'))

    print("running time {:.4f}".format(time()-t))
    #marginals = bpgc.return_marginals()
    #print(np.array(marginals))
    return 0

if __name__=='__main__':
    main()
