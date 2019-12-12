import numpy as np
import itertools
import modbp
import multilayerGM as gm
import os,re,sys
import shutil
import scipy.io as scio
from subprocess import Popen,PIPE
# clusterdir=os.path.abspath('../..') # should be in test/multilayer_benchmark_matlab
clusterdir=os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))



#main file for alling matlab

#shell scripts for calling matlab functions from command line
call_genlouvain_file = os.path.join(clusterdir,"test/genlouvain_mlsbm/call_matlab_genlouvain.sh")

#set architecture flag for compiled files
oncluster=False
if re.search("/nas/longleaf",clusterdir):
    oncluster=True
arch = "elf64" if oncluster else "x86_64" #for different compiled code to run


matlabbench_dir=os.path.join(clusterdir, 'test/multilayer_benchmark_matlab/')
matlaboutdir = os.path.join(matlabbench_dir,"matlab_temp_outfiles")
if not os.path.exists(matlaboutdir):
    os.makedirs(matlaboutdir)
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


def convert_nxmg_to_mbp_multigraph(nxmg,multiplex=True):
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
        if multiplex:
            for ind1, ind2 in itertools.combinations(curnodes, 2):
                interelist.append((ind1, ind2))
        else: #temporal - use only the next layer's node
            for k,ind1 in enumerate(curnodes[:-1]):
                ind2=curnodes[k+1]
                interelist.append((ind1, ind2))

    partition = list(nxmg.nodes(data='mesoset'))

    partition = list(map(lambda x: (node_inds[x[0]], x[1]), partition))
    partition = sorted(partition, key=lambda x: x[0])
    comvec = [x[1] for x in partition]
    return modbp.MultilayerGraph(comm_vec=comvec, interlayer_edges=interelist,
                                 intralayer_edges=intraelist,
                                 layer_vec=layervec)

#THIS HASN"T BEEN IMPLEMENTED IN PYTHON YET
# def create_multiplex_graph_block(n_nodes=100, n_layers=5, mu=.99,
#                                  p_in=.8,p_out=0,nblocks=3, maxcoms=10, k_max=150,k_min=3):
#     theta = 1
#     dt = gm.dependency_tensors.BlockMultiplex(n_nodes, n_layers, nblocks=nblocks,p_in=p_in,p_out=p_out)
#     null = gm.dirichlet_null(layers=dt.shape[1:], theta=theta, n_sets=maxcoms)
#     partition = gm.sample_partition(dependency_tensor=dt, null_distribution=null)
#     # with use the degree corrected SBM to mirror paper
#     multinet = gm.multilayer_DCSBM_network(partition, mu=mu, k_min=k_min, k_max=k_max, t_k=-2)
#     #     return multinet
#     #the multiplex connections should be the same here
#     mbpmulltinet = convert_nxmg_to_mbp_multigraph(multinet)
#     return mbpmulltinet

def create_temporal_graph(n_nodes=100, n_layers=5, mu=.99, p=.1, ncoms=5, k_max=30,k_min=3):
    theta = 1
    dt = gm.dependency_tensors.Temporal(n_nodes, n_layers, p)
    null = gm.dirichlet_null(layers=dt.shape[1:], theta=theta, n_sets=ncoms)
    partition = gm.sample_partition(dependency_tensor=dt, null_distribution=null)

    # with use the degree corrected SBM to mirror paper
    multinet = gm.multilayer_DCSBM_network(partition, mu=mu, k_min=k_min, k_max=k_max, t_k=-2)
    #     return multinet
    mbpmulltinet = convert_nxmg_to_mbp_multigraph(multinet,multiplex=False)
    return mbpmulltinet


def create_multiplex_graph(n_nodes=100, n_layers=5, mu=.99, p=.1, ncoms=10, k_max=150,k_min=3):
    theta = 1
    dt = gm.dependency_tensors.UniformMultiplex(n_nodes, n_layers, p)
    null = gm.dirichlet_null(layers=dt.shape[1:], theta=theta, n_sets=ncoms)
    partition = gm.sample_partition(dependency_tensor=dt, null_distribution=null)

    # with use the degree corrected SBM to mirror paper
    multinet = gm.multilayer_DCSBM_network(partition, mu=mu, k_min=k_min, k_max=k_max, t_k=-2)
    #     return multinet
    mbpmulltinet = convert_nxmg_to_mbp_multigraph(multinet,multiplex=True)
    return mbpmulltinet


#For the block multiplex we have to use the matlab
# since the dependency matrix hadn't been implemented in python
#at the time of running
def create_multiplex_graph_matlab(n_nodes=1000, nlayers=15, mu=.99,nblocks=3,p_in=.9,p_out=0,ismultiplex = True, ncoms=2):
    rprefix=np.random.randint(1000000)
    rprefix_dir=os.path.join(matlaboutdir,str(rprefix))
    if not os.path.exists(rprefix_dir):
        os.makedirs(rprefix_dir)

    moutputfile=os.path.join(rprefix_dir,'network.mat')

    parameters = [call_matlab_createbenchmark_file,
                  moutputfile,
                  "{:d}".format(n_nodes),
                  "{:d}".format(nlayers),
                  "{:d}".format(nblocks),
                  "{:.5f}".format(mu),
                  "{:.5f}".format(p_in),
                  "{:.5f}".format(p_out),
                  "{:d}".format(ncoms)
                  ]
    print(parameters)
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("creating benchmark graph failed : {:}".format(stderr))

    print(stderr)

    mlgraph=create_ml_graph_from_matlab(moutputfile,ismultiplex=ismultiplex)

    #clean out random graph
    if os.path.exists("{:}".format(rprefix_dir)):
        shutil.rmtree("{:}".format(rprefix_dir))

    return mlgraph

def call_gen_louvain(mgraph, gamma, omega, S=None):
    A, C = mgraph.to_scipy_csr()
    P = mgraph.create_null_adj()
    # print(A.shape,C.shape,P.shape)

    rprefix = np.random.randint(100000)
    scio_outfile = os.path.join(matlaboutdir, "{:d}_temp_matlab_input_file.mat".format(rprefix))
    matlaboutput = os.path.join(matlaboutdir, "{:d}_temp_matlab_output_file.mat".format(rprefix))
    T=mgraph.nlayers

    if S is None:
        scio.savemat(scio_outfile, {"A": A, "C": C, "P": P,"T":T})
    else:

        scio.savemat(scio_outfile, {"A": A, "C": C, "P": P,"T":T,
                                    "S0": np.reshape(S, (-1, mgraph.nlayers)).astype(float)})  # add in starting vector
    parameters = [call_genlouvain_file,
                  scio_outfile,
                  matlaboutput,
                  "{:.4f}".format(gamma),
                  "{:.4f}".format(omega)
                  ]
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    if process.returncode != 0:
        print("matlab call failed")
    print(stderr)

    try:
        S = scio.loadmat(matlaboutput)['S'][:, 0]
    except:
        print(stderr)
        os.remove(scio_outfile)
        raise (AssertionError,"matlab failed to run. can't find output file") #this should still in intercepted below

    try:
        os.remove(scio_outfile)
    except:
        pass
    try:
        os.remove(matlaboutput)
    except:
        pass

    return S


def run_ZMBP_on_graph(graph, q, beta,niters=100):

    sbmbpfile = os.path.join(clusterdir,'test/compare_with_ZM/modbp/mod')
    outdir = os.path.join(clusterdir,'test/compare_with_ZM/zm_outdir')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tmp_grph_file = os.path.join(outdir, 'temporary_graph_file.gml')
    graph.save(tmp_grph_file)

    parameters = [
        sbmbpfile, 'infer',
        "-l", tmp_grph_file,  # graph file
        "-v", "{:d}".format(5),
        '-t', "{:d}".format(niters),
        '-q', '{:d}'.format(q),
        '-b', '{:.6f}'.format(beta),
        '--confi', '{:}_q{:d}_marginals.txt'.format(tmp_grph_file, q),  # outfile
        '-M', '{:}_q{:d}_marginals.txt'.format(tmp_grph_file, q),  # outfile
        '-d', '1',  # use degree corrected
        '-i', '1'  # initialize randomly
    ]
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
    stdout=str(stdout)
    if process.returncode != 0:
        raise RuntimeError("running ZM_BP failed : {:}".format(stderr))

    srch = re.search("(?<=iter_time=)\d+", stdout)
    if srch:
        niters = int(srch.group())
    else:
        raise AssertionError("number iterations not found")
    # this is the file where the marginals are stored ( i.e what community is most
    # likely for each node)
    marginal_file = '{:}_q{:d}_marginals.txt'.format(tmp_grph_file, q)
    marginals = []

    with open(marginal_file, 'r') as f:
        inmargs = False

        for i, line in enumerate(f.readlines()):
            if re.search("^marginals:", line):
                inmargs = True
                continue
            if inmargs:
                if not re.search("\d+", line):
                    inmargs = False
                else:
                    marginals.append(list(map(lambda x: float(x),line.split())))

    marginals = np.array(marginals, dtype=float)
    return niters, marginals