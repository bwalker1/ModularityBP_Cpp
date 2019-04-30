from __future__ import division
from context import modbp
from time import time
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
import os, gzip, pickle
import sklearn.metrics as skm

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
        c =(cin + cout)
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
    import numpy as np
    import modbp
    import matplotlib.pyplot as plt
    import igraph as ig
    import seaborn as sbn
    #import network_tools as nt

    n = 1000
    q = 4
    ep = .15
    c = 6

    sbm = modbp.GenerateGraphs.generate_planted_partitions_sbm(n=n, c=c, epsilon=ep, ncoms=q)
    mbp_obj = modbp.ModularityBP(mlgraph=sbm, accuracy_off=True,
                                 align_communities_across_layers=True)

    omega = 0
    gamma = 1
    qmax = 6
    qrange = range(2, qmax + 1)
    betas_stars = [mbp_obj.get_bstar(omega=0, q=q) for q in qrange]
    betas = np.linspace(0, mbp_obj.get_bstar(omega=0, q=qmax), 20)
    betas = np.linspace(0, 6, 100)
    for gamma in [.5, 1, 1.5]:
        for j, beta in enumerate(betas):
            if j % 10 == 0:
                print(gamma, j, beta)
            for i in range(20):
                #     print(beta)
                mbp_obj.run_modbp(q=qmax, beta=beta, niter=500, omega=omega, resgamma=gamma,
                                  reset=True)

    rm_df = mbp_obj.retrieval_modularities
    return rm_df

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
    print ("Running for beta=0.01")
    mbpinterface.run_modbp(q=2, beta=0.01,resgamma=.8)
    #mbpinterface.run_modbp(q=2, beta=.1)
    #mbpinterface.run_modbp(q=2, beta=.2)
    print ("Running for beta=1.2")
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

def test_retmod_calculation():
    n=1000
    q=2
    nblocks=q
    c=3
    ep=.2
    pin=c/(1.0+ep*(q-1.0))/(n*1.0/q)
    pout=c/(1+(q-1.0)/ep)/(n*1.0/q)
    prob_mat = np.identity(q) * pin + (np.ones((q, q)) - np.identity(q)) * pout
    print (prob_mat    )
    g=ig.load('notworking2com_graph.graphml.gz')
    randSBM=modbp.RandomSBMGraph(n,prob_mat,graph=g)
    pinterface=modbp.ModularityBP(randSBM.graph)
    mbpinterface.run_modbp(beta=.01,q=2,niter=500)
    vc=ig.VertexClustering(graph=g,membership=mbpinterface.partitions[0])
    print("ig VC modularity: {:.7f}".format(vc.modularity))
    print("calculated modularity : {:.7f}".format(mbpinterface.retrieval_modularities['retrieval_modularity'][0]))    

def test_generate_graph():
    # np.random.seed(1)
    n = 1000
    q = 2
    nlayers=1
    nblocks = q
    c = 10
    ep = .01
    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout    # print()    
    ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=0)
    mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.interedges, ml_sbm.layer_vec,                  comm_vec=ml_sbm.get_all_layers_block())
    mlbp = modbp.ModularityBP(mlgraph=mgraph)

    bstar=mlbp.get_bstar(q=q)
    beta = bstar
    mlbp.run_modbp(beta=beta, resgamma=1, q=q,niter=500,omega=0)


    #for beta in np.linspace(1.5,2,15):
        #mlbp.run_modbp(beta=beta, resgamma=1, q=q,niter=100,omega=0)
        #print("AMI beta:{:.2f} =  {:.3f}".format(beta,mgraph.get_AMI_with_communities(mlbp.partitions[q][beta])))
    # pmat=get_partition_matrix(mlbp.partitions[3][1],mlbp.layer_vec)

def test_modbp_interface():
    # confirmt that it is still working on the single layer case
    
    n = 100
    q = 2
    nlayers = 1
    nblocks = q
    c = 1
    ep = .04
    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=.1)
    mgraph = modbp.MultilayerGraph(interlayer_edges=ml_sbm.intraedges,
                                   intralayer_edges=ml_sbm.interedges, layer_vec=ml_sbm.layer_vec,
                                   comm_vec=ml_sbm.get_all_layers_block())
    mlbp = modbp.ModularityBP(mlgraph=mgraph, accuracy_off=True, use_effective=False)
    inferbp = modbp.InferenceBP(mlgraph=mgraph)
    mlbp.run_modbp(beta=0, resgamma=1, q=q,niter=1000,omega=0)
    #inferbp.run_modbp(q=q,niter=1000)
    # betas=np.linspace(.5,2.5,50)return
    return
    ntrials = 1
    qvals = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    bstars = map(lambda x: mlbp.get_bstar(x), qvals)

    # resgammas=np.linspace(.1,1.5,5)
    resgammas = np.array([.5, 1, 1.5])
    for i, q in enumerate(qvals):
        print('trial {:d}'.format(q))
        betas = np.linspace(bstars[i] - .2, bstars[i] + .2, 10)
        for j, beta in enumerate(betas):
            mlbp.run_modbp(q=q, beta=beta, resgamma=1.0, omega=0, niter=1000, reset=True)
            #print mlbp.retrieval_modularities

def test_community_swapping_ml():
    n = 250
    q = 2
    nlayers = 20
    eta = 0.0
    c = 10
    ep = .1
    ntrials = 1
    omega = 2.0
    gamma = .5
    nblocks = q

    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    output = pd.DataFrame(columns=['ep', 'eta', 'beta', 'resgamma', 'omega', 'niters',
                                   'AMI', 'AMI_layer_avg', 'retrieval_modularity', 'bethe_free_energy',
                                   'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial'])

    qmax = 2 * q
    for trial in range(ntrials):
        ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
        mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.layer_vec, ml_sbm.interedges,
                                       comm_vec=ml_sbm.get_all_layers_block())

        #add

        mlbp = modbp.ModularityBP(mlgraph=mgraph, use_effective=True, accuracy_off=False,
                                    align_communities_across_layers=True)

        bstars = [mlbp.get_bstar(q_, omega) for q_ in range(2, qmax + 1)]
        betas = np.linspace(bstars[0], bstars[-1], 3 * len(bstars))
        bstar = mlbp.get_bstar(qmax, omega)
        for beta in [bstar]: #just run at bstar.
            mlbp.run_modbp(beta=beta, niter=2000, q=qmax, resgamma=gamma, omega=omega)

            print("Group mapping")
            print(mlbp.marginal_index_to_close_marginals[0])
            print(mlbp._groupmap_to_permutation_vector(0))

            # print(mlbp.marginal_to_comm_number[0])
            # print(mlbp._groupmap_to_permutation_vector(0))
            # print('Test permutation sweep.')
            # mlbp_rm = mlbp.retrieval_modularities
            # old_part=mlbp.partitions[0].copy() #before change
            # old_transformed=np.zeros(len(old_part))
            # mlbp._perform_permuation_sweep(0) # permute all layers
            # print('AMI after permutation sweep')
            # print(skm.adjusted_mutual_info_score(old_part,mlbp.partitions[0]))
            # # print('old',old_part)
            # # print('new',mlbp.partitions[0])
            # for layer in mlbp.layers_unique:
            #     cinds=np.where(mlbp.layer_vec==layer)[0]
            #     old_transformed[cinds]=map( lambda x : mlbp._permutation_vectors[0][layer][x] ,old_part[cinds])
            # print('AMI of old transformed with new ',
            #       skm.adjusted_mutual_info_score(old_transformed,mlbp.partitions[0]))
            plt.close()
            f, a = plt.subplots(1, 2, figsize=(6, 3))
            a = plt.subplot(1, 2, 1)
            mlbp.plot_communities(ax=a)
            a = plt.subplot(1, 2, 2)
            mlbp.plot_communities(0, ax=a)
            plt.show()


def test_cpp_permutation():
    n = 200
    q = 2
    nlayers = 20
    eta = 0.01
    c = 10
    ep = .1
    ntrials = 1
    omega = 2.0
    gamma = .5
    nblocks = q

    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    output = pd.DataFrame(columns=['ep', 'eta', 'beta', 'resgamma', 'omega', 'niters',
                                   'AMI', 'AMI_layer_avg', 'retrieval_modularity', 'bethe_free_energy',
                                   'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial'])

    qmax = 4

    ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
    mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.layer_vec, ml_sbm.interedges,
                                   comm_vec=ml_sbm.get_all_layers_block())
    weights=[1.0 for _ in range(len(mgraph.intralayer_edges))]
    mgraph.interlayer_weights=weights
    mlbp = modbp.ModularityBP(mlgraph=mgraph, use_effective=True, accuracy_off=False,
                              align_communities_across_layers=False)

    bstar = mlbp.get_bstar(qmax, omega)

    mlbp.run_modbp(beta=bstar, niter=2000, q=qmax, resgamma=gamma, omega=omega)
    #after first run with no realignment
    plt.close()
    f, a = plt.subplots(1, 1)
    mlbp.plot_communities(0, ax=a)
    plt.show()
    mlbp._perform_permuation_sweep(ind=0)
    mlbp._switch_beliefs_bp(0)
    # #after realignment
    plt.close()
    f, a = plt.subplots(1, 1)
    mlbp.plot_communities(0, ax=a)
    plt.show()



    #
    #this should just flip the marginals
    tmap={0:1,1:0}
    out=np.array([ np.array(map(lambda x: tmap[x] ,range(2))) for _ in range(mlbp.nlayers)])
    cmargs = np.array(mlbp._bpmod.return_marginals())
    mlbp.marginals[0] = cmargs
    # Calculate effective group size and get partitions
    mlbp._get_community_distances(0)  # sets values in method
    cpartition = mlbp._get_partition(0, True)
    mlbp.partitions[0] = cpartition
    plt.close()
    f, a = plt.subplots(1, 1)
    mlbp.plot_communities(0, ax=a)
    plt.show()


def test_bethe_free_energy_calc():
    n = 1000
    q = 2
    nlayers = 20
    eta = 0.01
    c = 10
    ep = .1
    ntrials = 1
    omega = 1.0
    gamma = 1.0
    nblocks = q
    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    output = pd.DataFrame(columns=['beta', 'resgamma', 'omega', 'niters',
                                   'AMI', 'AMI_layer_avg', 'retrieval_modularity', 'bethe_free_energy',
                                   'Accuracy', 'Accuracy_layer_avg', 'qstar', 'num_coms', 'is_trivial'])

    qmax = 4
    ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
    mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.layer_vec, ml_sbm.interedges,
                                   comm_vec=ml_sbm.get_all_layers_block())

    mlbp = modbp.ModularityBP(mlgraph=mgraph, use_effective=True, accuracy_off=False,
                              align_communities_across_layers=True)
    bstar = mlbp.get_bstar(qmax, omega)
    print('running')
    mlbp.run_modbp(beta=bstar, niter=4, q=qmax, resgamma=gamma, omega=omega,reset=False)

    for i in range(20):
        print(i)
        cind=output.shape[0]
        # output.loc[cind,:] = mlbp.retrieval_modularities.loc[mlbp.nruns,:]
        output.loc[cind,'bethe_free_energy']=mlbp._get_bethe_free_energy()
        output.loc[cind,'niters']=i
        mlbp.run_modbp(beta=bstar, niter=10, q=qmax, resgamma=gamma, omega=omega, reset=False)

    plt.close()
    plt.scatter(output.loc[:,'niters'],output.loc[:,'bethe_free_energy'])
    plt.show()


def test_spinglass():
    n = 200
    c = 5
    p = (n * c / 2.0) / (n * (n - 1.0) / 2.0)
    qmax = 2
    omega = 0
    gamma = 1
    er_graph = modbp.RandomERGraph(n=200, p=p)
    betas = np.linspace(50, 65, 10)
    mler_graph = modbp.MultilayerGraph(intralayer_edges=er_graph.get_edgelist(),
                                       layer_vec=[0] * er_graph.n)
    mbp_obj = modbp.ModularityBP(mlgraph=mler_graph, accuracy_off=True,
                                 align_communities_across_layers=True)

    for gamma in [1]:
        for j, beta in enumerate(betas):
            for i in range(1):
                mbp_obj.run_modbp(q=qmax, beta=beta, niter=400, omega=omega, resgamma=gamma,
                                  reset=True)

    rm_df = mbp_obj.retrieval_modularities
    niters = rm_df.astype(float).groupby(['beta'])['niters'].mean()

    plt.close()
    f, a = plt.subplots(1, 1, figsize=(5, 5))
    a = plt.subplot(1, 1, 1)
    a.plot(niters.index, niters.values)
    plt.show()

def main():
    test_spinglass()


if __name__=='__main__':
    main()
