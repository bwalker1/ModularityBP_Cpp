from context import modbp
from time import time
import numpy as np

def main():
    n=100000
    t=time()
    erg=modbp.RandomERGraph(n=n,p=3.0/n)
    print("time to construct {:.4f}".format(time()-t))
    elist=erg.get_edgelist()
    elist.sort()
    pv=modbp.bp.PairVector(elist)
    bpgc=modbp.BP_Modularity(n=n, p=3.0 / n, q=4, beta=1, transform=False, simultaneous=False)
    t=time()
    bpgc.run()
    print("running time {:.4f}".format(time()-t))
    bpgc.print_marginals(limit=10)
    t=bpgc.return5(pv)
    return 0

if __name__=='__main__':
    main()