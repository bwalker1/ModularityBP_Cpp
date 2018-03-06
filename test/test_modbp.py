from context import modbp
from time import time
import numpy as np

def main():
    n=1000
    t=time()

    prob_mat=np.array([[.7,.1],[.1,.7]])
    #erg=modbp.RandomSBMGraph(n=n,comm_prob_mat=prob_mat)
    erg = modbp.RandomERGraph(n=n,p=3.0/n)
    print("time to construct {:.4f}".format(time()-t))
    elist=erg.get_edgelist()
    elist.sort()
    pv=modbp.bp.PairVector(elist)
    bpgc=modbp.BP_Modularity(edgelist=pv, _n=n, q=4, beta=1, transform=False)
    t=time()
    bpgc.run()
    print( "BFE={:.3f}".format(bpgc.compute_bethe_free_energy()))
    print("running time {:.4f}".format(time()-t))
    marginals = bpgc.return_marginals()
    print(np.array(marginals))
    return 0

if __name__=='__main__':
    main()
