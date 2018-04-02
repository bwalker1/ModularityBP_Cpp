from __future__ import division
from context import modbp
from time import time
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd

def test_detection():
    n=10000
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
    print (prob_mat)

    read = False #was using the same graph everytime for testing.
    if read:
        print ('loading graph from file')
        RSBM = modbp.RandomSBMGraph(n, prob_mat, graph=ig.load('RSMB_test.graphml.gz'))
        print ("{:d},{:d}".format(RSBM.n,RSBM.m))
    else:
        RSBM = modbp.RandomSBMGraph(n=n, comm_prob_mat=prob_mat)
        RSBM.graph.save('RSMB_test.graphml.gz')
    randSBM = modbp.RandomSBMGraph(n, prob_mat)

    mbpinterface = modbp.ModularityBP(randSBM.graph)
    mbpinterface.run_modbp(q=2, beta=1.2)
    #mbpinterface.run_modbp(q=2, beta=.2)
    #mbpinterface.run_modbp(q=2, beta=.1)
    mbpinterface.run_modbp(q=2, beta=.01)
    print ("When run first, %f"%mbpinterface.retrival_modularities[2][1.2])
    #print("")
    mbpinterface = modbp.ModularityBP(randSBM.graph)
    print "Running for beta=0.01"
    mbpinterface.run_modbp(q=2, beta=0.01,resgamma=.8)
    #mbpinterface.run_modbp(q=2, beta=.1)
    #mbpinterface.run_modbp(q=2, beta=.2)
    print "Running for beta=1.2"
    mbpinterface.run_modbp(q=2, beta=1.2,resgamma=.8)
    
    print("When run last, %f"%mbpinterface.retrival_modularities[2][1.2])

    # marg = np.array(bpgc.return_marginals())
    # print (marg[:5])
    # part=np.argmax(marg,axis=1)
    # print ('niters to converge', bpgc.run(1000))
    # print ("AMI: {:.3f}".format(RSBM.get_AMI_with_blocks(labels=part)))
    # print ("percent: {:.3f}".format(np.sum(RSBM.block == part) / (1.0 * n)))
    # #test it with the calss method
    # mbpinterface = modbp.ModularityBP(RSBM.graph)  # create class
    # mbpinterface.run_modbp(beta,2,1000)
    # print(mbpinterface.marginals[2][beta][:5])
    # print ('niters to converge',mbpinterface.niters[2][beta])
    # print ('modularity: {:.4f}'.format(mbpinterface.retrival_modularities[2][beta]))
    # print 'AMI=',RSBM.get_AMI_with_blocks(mbpinterface.partitions[2][beta])
    # print "accuracy=",RSBM.get_accuracy(mbpinterface.partitions[2][beta])

def test_fbnetwork():
    fbnet = ig.load("./football.net.graphml.gz")
    mbpinter = modbp.ModularityBP(fbnet)

    # qs=np.arange(4,15)
    qs = np.array([7, 8, 9, 10])
    colors = sbn.cubehelix_palette(n_colors=len(qs))
    gammas = np.linspace(.5, 1.5, 10)
    # gammas=np.array([,1.1])

    pd.DataFrame()
    for gam in gammas:
        for q in qs:
            bstar = mbpinter.get_bstar(q)
            #         betas=np.linspace(bstar-.25,bstar+.25,10)
            betas = np.array([bstar])

            #         betas=np.linspace(0,2.5,100)
            for beta in betas:
                mbpinter.run_modbp(q=q, beta=beta, resgamma=gam, niter=500)
    return 0

def test_generate_graph():
    np.random.seed(1)
    n = 30
    nlayers = 3
    q = 3
    nblocks = q
    c = 5.0
    ep = .001
    pin = c / (1.0 + ep) / (n * 1.0 / q)
    pout = c / (1 + 1.0 / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout    # print()
    # print()
    # print(ml_sbm.layer_sbms[0].graph.vs['id'])
    # print(ml_sbm.layer_sbms[0].graph.vs['block'])
    # print()
    # print(ml_sbm.layer_sbms[1].graph.vs['id'])
    # print(ml_sbm.layer_sbms[1].graph.vs['block'])
    # print()
    # print(ml_sbm.layer_sbms[2].graph.vs['id'])
    # print(ml_sbm.layer_sbms[2].graph.vs['block'])
    #create a multigraph from the MLSBM
    ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=.2)
    mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.interedges, ml_sbm.layer_vec)
    print([g.vcount() for g in mgraph.layers])
    mlbp = modbp.ModularityBP(mlgraph=mgraph)
    mlbp.run_modbp(beta=1, resgamma=1, q=3)
    print('stop')

def main():
    test_generate_graph()
if __name__=='__main__':
    main()
