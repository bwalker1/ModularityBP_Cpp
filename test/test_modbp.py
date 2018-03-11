from __future__ import division
from context import modbp
from time import time
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

def test_detection():
    n=1000000
    q=4
    nblocks=q
    cmin = 1
    cmax = 10
    pin = 5*q/n
    pout = 0.5*q/n
    prob_mat=np.identity(nblocks)*pin + (np.ones((nblocks,nblocks))-np.identity(nblocks))*pout
    print (prob_mat)
    RSBM = modbp.RandomSBMGraph(n=n,comm_prob_mat=prob_mat)
    ER = modbp.RandomERGraph(n=n,p=5/n)
    print (np.array(ER.get_edgelist()))
    return

    step = (cmax/cmin)/11
    nsamples = len(np.arange(cmin,cmax,step))
    xs = np.empty(nsamples)
    ys = np.empty(nsamples)
    count = 0
    for cin in np.arange(cmin,cmax,step):
		cout = 1;
		c = (cin + cout)
		beta = np.log(q/(np.sqrt(c)-1) + 1)
		pin = cin/(n/q);
		pout= cout/((q-1)*n/q);
		#print "%f %f"%(pin,pout)
		t=time()
		prob_mat=np.identity(nblocks)*pin + (np.ones((nblocks,nblocks))-np.identity(nblocks))*pout

		RSBM = modbp.RandomSBMGraph(n=n,comm_prob_mat=prob_mat)
		m= RSBM.m

	#print("time to construct {:.4f}".format(time()-t))
		elist=RSBM.get_edgelist()
		elist.sort()
		pv=modbp.bp.PairVector(elist)
		bpgc=modbp.BP_Modularity(edgelist=pv, _n=n, q=q, beta=beta, transform=False)
		#old_marg=np.array(bpgc.return_marginals())
		#for i in range(10):
		#    bpgc.step()
		#    new_marg=np.array(bpgc.return_marginals())
		#    print ("Change in margins {:d}: {:.3f}".format(i,np.sum(np.abs(old_marg-new_marg))/(1.0*q*n)))
		#    old_marg=new_marg
		bpgc.run()
		marg = bpgc.return_marginals()
	
		color_dict={0:"red",1:"blue",2:'green',3:"magenta"}
		RSBM.graph.vs['color']=map(lambda x : color_dict[np.argmax(x)],marg)
		ami = RSBM.get_AMI_with_blocks(RSBM.graph.vs['color'])
		print("NMI: {:.3f}".format(ami))
		xs[count] = cin
		ys[count] = ami
		count += 1
		#ig.plot(RSBM.graph,layout=RSBM.graph.layout('kk'))

    print("running time {:.4f}".format(time()-t))
    #marginals = bpgc.return_marginals()
    #print(np.array(marginals))
    plt.plot(xs,ys)
    plt.show()
    return 0
    
def test_transform():
	n=100000
	q=4
	nblocks=q

	cin = 5
	cout = 1
	c = (cin + cout)

	beta = np.log(q/(np.sqrt(c)-1) + 1)
	pin = cin/(n/q);
	pout= cout/((q-1)*n/q);
	#print "%f %f"%(pin,pout)
	t=time()
	prob_mat=np.identity(nblocks)*pin + (np.ones((nblocks,nblocks))-np.identity(nblocks))*pout

	RSBM = modbp.RandomSBMGraph(n=n,comm_prob_mat=prob_mat)
	m= RSBM.m

	elist=RSBM.get_edgelist()
	elist.sort()
	pv=modbp.bp.PairVector(elist)
	bpgc=modbp.BP_Modularity(edgelist=pv, _n=n, q=q, beta=beta, transform=False)
	
	t = time()
	bpgc.run()
	print("running time {:.4f}".format(time()-t))
	
def test_qstar():
	pass

def test_modinterface_class():
    n=200
    q=3
    pin = 5 * q / n
    pout = 0.5 * q / n
    prob_mat = np.identity(q) * pin + (np.ones((q, q)) - np.identity(q)) * pout
    randSBM=modbp.RandomSBMGraph(n,prob_mat)
    mbpinterface=modbp.ModularityBP(randSBM.graph)
    mbpinterface.run_modbp(beta=1.0,q=q)
    # mbpinterface.run_modbp(beta=.5,q=2)
    print (mbpinterface.retrival_modularities)

def main():
	#test_transform()
    test_modinterface_class()

if __name__=='__main__':
    main()
