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
    n = 1000
    q = 2
    nblocks = q
    c = 3.0
    ep = .1
    pin = c / (1.0 + ep) / (n * 1.0 / q)
    pout = c / (1 + 1.0 / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    read = True
    if read:
        print 'loading graph from file'
        RSBM = modbp.RandomSBMGraph(n, prob_mat, graph=ig.load('RSMB_test.graphml.gz'))
        print ("{:d},{:d}".format(RSBM.n,RSBM.m))
    else:
        RSBM = modbp.RandomSBMGraph(n=n, comm_prob_mat=prob_mat)
        RSBM.graph.save('RSMB_test.graphml.gz')

    beta=1.3
    #call directly
    elist = RSBM.get_edgelist()
    elist.sort()
    pv = modbp.bp.PairVector(elist)
    bpgc = modbp.BP_Modularity(edgelist=pv, _n=n, q=q, beta=beta, transform=False)


    marg = np.array(bpgc.return_marginals())
    print (marg[:5])
    part=np.argmax(marg,axis=1)
    print ('niters to converge', bpgc.run(1000))
    print ("AMI: {:.3f}".format(RSBM.get_AMI_with_blocks(labels=part)))
    print ("percent: {:.3f}".format(np.sum(RSBM.block == part) / (1.0 * n)))

    #test it with the calss method
    mbpinterface = modbp.ModularityBP(RSBM.graph)  # create class
    mbpinterface.run_modbp(beta,2,1000)
    print(mbpinterface.marginals[2][beta][:5])
    print ('niters to converge',mbpinterface.niters[2][beta])
    print ('modularity: {:.4f}'.format(mbpinterface.retrival_modularities[2][beta]))
    print 'AMI=',RSBM.get_AMI_with_blocks(mbpinterface.partitions[2][beta])
    print "accuracy=",RSBM.get_accuracy(mbpinterface.partitions[2][beta])

# def test_generategraph():
#     t=time()
#     RER=modbp.RandomERGraph(n=1000,p=.05)
#     print (RER.edgelist[:5])
#     print ('creationg time',time()-t)

def main():
	#test_transform()
    test_modinterface_class()

if __name__=='__main__':
    main()
