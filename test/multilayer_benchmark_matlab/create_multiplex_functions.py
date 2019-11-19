import numpy as np
import itertools
import modbp
import multilayerGM as gm
import os
import shutil
import scipy.io as scio
from subprocess import Popen,PIPE
clusterdir=os.path.abspath('../..') # should be in test/multilayer_benchmark_matlab
matlabbench_dir=os.path.join(clusterdir, 'test/multilayer_benchmark_matlab/')
matlaboutdir = os.path.join(matlabbench_dir,"matlab_temp_outfiles")
call_matlab_createbenchmark_file = os.path.join(matlabbench_dir, "call_matlab_multilayer.sh")


def adjacency_to_edges(A,offset=0):
    nnz_inds = np.nonzero(A)
    nnzvals = np.array(A[nnz_inds])
    if len(nnzvals.shape) > 1:
        nnzvals = nnzvals[0]  # handle scipy sparse types
    return list(zip(nnz_inds[0]+offset, nnz_inds[1]+offset, nnzvals))

def create_ml_graph_from_matlab(moutputfile,ismultiplex=True):
    matoutputdict=scio.loadmat(moutputfile)
    A=matoutputdict['A']
    nnodes=A[0][0].shape[0]
    nlayers=len(A)
    layer_vec=[ i//nnodes for i in range(nnodes*nlayers)]
    for i,A in enumerate(A):
        if i==0:
            all_intra_edges=adjacency_to_edges(A[0])
        else:
            all_intra_edges+=adjacency_to_edges(A[0],offset=i*nnodes)

    interlayer_edges=[]
    for i in range(nnodes):
        if ismultiplex:
            #connect all possible pairs state nodes
            interlayer_edges.extend( itertools.combinations([i+l*nnodes for l in range(nlayers)],2))
        else: #temporal case only connect adjacent
            interlayer_edges.extend([ (i+(l*nnodes), i+(l+1)*nnodes) for l in range(1,nlayers-1)])

    comm_vec=matoutputdict['S'].flatten('F') #flatten by column


    mlgraph=modbp.MultilayerGraph(intralayer_edges=all_intra_edges,interlayer_edges=interlayer_edges,\
                                  layer_vec=layer_vec,comm_vec=comm_vec)

    return mlgraph


def convert_nxmg_to_mbp_multigraph(nxmg):
    # dt has the interlayer edges in it
    nodelist = np.array(list(nxmg.adj.keys()))
    layervec = nodelist[:, 1]
    N = len(nodelist)
    layers, layercounts = np.unique(layervec, return_counts=True)
    assert (len(np.unique(layercounts)) == 1), "Multiplex must have same number of edges in each layer"
    nodeperlayer = layercounts[0]

    layer_adjust_ind_dict = dict(zip(layers, np.append([0], np.cumsum(layercounts)[:-1])))
    node_inds = dict([((n, lay), layer_adjust_ind_dict[lay] + n) for n, lay in nodelist])
    interelist = []
    intraelist = []
    edges = np.array(nxmg.edges)
    # seperate edges by type
    for e1, e2 in edges:
        ind1 = node_inds[(e1[0], e1[1])]
        ind2 = node_inds[(e2[0], e2[1])]
        assert e1[1] == e2[1], "Non intralayer edges identified in multiplex"
        intraelist.append((ind1, ind2))

    # We create a multiplex interedge list here
    for i in range(nodeperlayer):  # i is node number
        curnodes = [i + j * (nodeperlayer) for j in range(len(layers))]
        for ind1, ind2 in itertools.combinations(curnodes, 2):
            interelist.append((ind1, ind2))

    partition = list(nxmg.nodes(data='mesoset'))

    partition = list(map(lambda x: (node_inds[x[0]], x[1]), partition))
    partition = sorted(partition, key=lambda x: x[0])
    comvec = [x[1] for x in partition]
    return modbp.MultilayerGraph(comm_vec=comvec, interlayer_edges=interelist,
                                 intralayer_edges=intraelist,
                                 layer_vec=layervec)


def create_multiplex_graph(n_nodes=100, n_layers=5, mu=.99, p=.1, maxcoms=10, k_max=150,k_min=3):
    theta = 1
    dt = gm.dependency_tensors.UniformMultiplex(n_nodes, n_layers, p)
    null = gm.dirichlet_null(layers=dt.shape[1:], theta=theta, n_sets=maxcoms)
    partition = gm.sample_partition(dependency_tensor=dt, null_distribution=null)

    # with use the degree corrected SBM to mirror paper
    multinet = gm.multilayer_DCSBM_network(partition, mu=mu, k_min=k_min, k_max=k_max, t_k=-2)
    #     return multinet
    mbpmulltinet = convert_nxmg_to_mbp_multigraph(multinet)
    return mbpmulltinet


#original mehtod used the matlab code.  have since switched to the python .
def create_multiplex_graph_matlab(n=1000, nlayers=40, mu=.99, p=.1,ismultiplex = False, ncoms=2):
    rprefix=np.random.randint(1000000)
    rprefix_dir=os.path.join(matlaboutdir,str(rprefix))
    if not os.path.exists(rprefix_dir):
        os.makedirs(rprefix_dir)

    moutputfile=os.path.join(rprefix_dir,'network.mat')

    parameters = [call_matlab_createbenchmark_file,
                  moutputfile,
                  "{:d}".format(n),
                  "{:d}".format(nlayers),
                  "{:.5f}".format(mu),
                  "{:.5f}".format(p),  #p is the prop of transmitting community label!
                  "{:d}".format(ncoms)
                  ]
    print(parameters)
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("creating benchmark graph failed : {:}".format(stderr))

    mlgraph=create_multiplex_graph(moutputfile,ismultiplex=ismultiplex)

    #clean out random graph
    if os.path.exists("{:}".format(rprefix_dir)):
        shutil.rmtree("{:}".format(rprefix_dir))

    return mlgraph