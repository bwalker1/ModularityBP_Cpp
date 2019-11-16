from __future__ import division
import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import matplotlib.pyplot as plt
import sys
from subprocess import Popen,PIPE
import re
import os
import shutil
import scipy.io as scio
import sklearn.metrics as skm
import itertools
#generative multilayer benchmark models (now in python)
from time import time

from create_multiplex_functions import create_multiplex_graph

#import infomap

clusterdir=os.path.abspath('../..') # should be in test/multilayer_benchmark_matlab
matlabbench_dir=os.path.join(clusterdir, 'test/multilayer_benchmark_matlab/')
matlaboutdir = os.path.join(matlabbench_dir,"matlab_temp_outfiles")

if not os.path.exists(matlaboutdir):
    os.makedirs(matlaboutdir)
#main file for alling matlab

#shell scripts for calling matlab functions from command line
call_genlouvain_file = os.path.join(clusterdir,"test/genlouvain_mlsbm/call_matlab_genlouvain.sh")
call_matlab_createbenchmark_file = os.path.join(matlabbench_dir, "call_matlab_multilayer.sh")

#set architecture flag for compiled files
oncluster=False
if re.search("/nas/longleaf",clusterdir):
    oncluster=True
arch = "elf64" if oncluster else "x86_64" #for different compiled code to run


infomap_dir = os.path.join(clusterdir, "test/Infomap")
infomap_file = os.path.join(infomap_dir, "Infomap")
infomapoutdir = os.path.join(matlabbench_dir, "infomap_outputs")


def create_tuple_indices(nodesperlayer, nlayers, rev=False):
    # here we assume equal number of nodes in every layer
    totN = nlayers * nodesperlayer
    outdict = dict(zip(range(totN), [(i, j) for j in range(1, nlayers + 1) for i in range(1, nodesperlayer + 1)]))
    if rev:
        return dict([(val, k) for k, val in outdict.items()])
    return outdict


def create_netfile(graph, filename):
    assert len(np.unique([l.vcount() for l in graph.layers])) == 1, "All layers must have same length"
    n = graph.layers[0].vcount()
    indict = create_tuple_indices(n, graph.nlayers)
    with open(filename, 'w') as fh:
        # layer node layer node [weight]
        alledges = np.append(graph.intralayer_edges, graph.interlayer_edges, axis=0)
        # alledges = graph.intralayer_edges
        #         fh.write("*Vertices {:d}\n".format(n))
        #         for i in range(1,n+1):
        #             fh.write('{:d} "node {:d}\n"'.format(i,i))
        fh.write("*Multiplex\n")
        fh.write('# layer node layer node [weight]\n')
        for e in alledges:
            w = 1 if len(e) < 3 else e[2]
            n1, l1 = indict[e[0]]
            n2, l2 = indict[e[1]]
            fh.write("{:d} {:d} {:d} {:d} {:.6f}\n".format(l1, n1, l2, n2, w))
    return filename


def create_netfile_singlelayer(graph, filename):
    assert len(np.unique([l.vcount() for l in graph.layers])) == 1, "All layers must have same length"
    n = graph.layers[0].vcount()
    with open(filename, 'w') as fh:
        # layer node layer node [weight]
        alledges = np.append(graph.intralayer_edges, graph.interlayer_edges, axis=0)
        #         alledges=graph.intralayer_edges
        fh.write("*Vertices {:d}\n".format(graph.N))
        for i in range(0, n):
            fh.write('{:d} "node {:d}"\n'.format(i, i))
        fh.write("*Edges {}\n".format(len(alledges)))
        for e in alledges:
            w = 1 if len(e) < 3 else e[2]
            fh.write("{:d} {:d} {:.6f}\n".format(e[0], e[1], w))
    return filename


def loadclusters(revdict, filename):
    outcluster = []
    clusters = pd.read_table(filename, sep=' ', skiprows=2, header=None)
    clusters.columns = ['layer', 'node', 'cluster', 'flow']
    clusters['nid'] = list(map(lambda x: revdict[(x[0], x[1])], zip(clusters['node'], clusters['layer'])))
    return clusters


def loadclusters_single(filename):
    outcluster = []
    clusters = pd.read_table(filename, sep=' ', skiprows=2, header=None)
    clusters.columns = ['node', 'cluster', 'flow']
    clusters['nid'] = clusters['node']
    return clusters


def call_infomap(graph, r):
    assert len(np.unique([l.vcount() for l in graph.layers])) == 1, "All layers must have same length"
    n = graph.layers[0].vcount()
    revindict = create_tuple_indices(n, graph.nlayers, rev=True)
    rprefix = np.random.randint(1000000)
    rprefix_dir = os.path.join(infomapoutdir, str(rprefix))
    if not os.path.exists(rprefix_dir):
        os.makedirs(rprefix_dir)
    networkfile = os.path.join(rprefix_dir, 'network.net')
    single_layer = r < 0

    if single_layer:
        clusteroutfile = re.sub(".net\Z", ".clu", networkfile)
        create_netfile_singlelayer(graph, networkfile)
        parameters = [infomap_file,
                      "-i", "pajek",
                      "--clu", "--tree", '--expanded',
                      '-z',
                      "{:}".format(networkfile),
                      "{:}".format(rprefix_dir),
                      ]
    else:
        clusteroutfile = re.sub(".net\Z", "_expanded.clu", networkfile)
        create_netfile(graph, networkfile)
        parameters = [infomap_file,
                      "-i", "multilayer",
                      "--clu", "--tree", '--expanded',
                      "--multilayer-relax-rate", "{:.5f}".format(r),
                      "{:}".format(networkfile),
                      "{:}".format(rprefix_dir),
                      ]
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("running infomap failed : {:}".format(stderr))
    if single_layer:
        outcluster = loadclusters_single(clusteroutfile)
    else:
        outcluster = loadclusters(revindict, clusteroutfile)
    cluster = np.array(([-1 for _ in range(graph.N)]))
    for ind in outcluster.index:
        cluster[outcluster.loc[ind, 'nid']] = outcluster.loc[ind, 'cluster']

    try:
        shutil.rmtree(rprefix_dir)
    except:
        pass

    return cluster

#These infomap methods just did not work on the cluster
# def create_infomap_net(infomapwrapper, multilayernet,single_layer=False):
#
#     net = infomapwrapper.network()
#     assert len(np.unique([l.vcount() for l in multilayernet.layers])) == 1, "All layers must have same length"
#     n = multilayernet.layers[0].vcount()
#
#     nodeinddict = create_tuple_indices(n, multilayernet.nlayers)
#
#     # add in intralayer edges
#     for e in multilayernet.intralayer_edges:
#         if len(e) > 2:
#             w = e[2]
#         else:
#             w = 1.0
#         e1 = e[0]
#         e2 = e[1]
#         n1, l1 = nodeinddict[e1]
#         n2, l2 = nodeinddict[e2]
#         net.addMultilayerIntraLink(l1, n1, n2, w)
#     # add in interlayer edges
#     if not single_layer:
#         for e in multilayernet.interlayer_edges:
#             if len(e) > 2:
#                 w = e[2]
#             else:
#                 w = 1.0
#             e1 = e[0]
#             e2 = e[1]
#             n1, l1 = nodeinddict[e1]
#             n2, l2 = nodeinddict[e2]
#             net.addMultilayerInterLink(l1, n1, l2, w)
#     net.generateStateNetworkFromMultilayerWithInterLinks()
#     return infomapwrapper


#had trouble getting this to work
# def run_infomap_python(graph, r=.1):
#     assert len(np.unique([l.vcount() for l in graph.layers]))==1,"All layers must have same length"
#     n=graph.layers[0].vcount()
#     infomapSimple = infomap.Infomap("--two-level")
#     if r<0:
#         single_layer=True
#     else:
#         single_layer=False
#     infomapSimple = create_infomap_net(infomapSimple, graph,single_layer=single_layer)
#     infomapSimple.multilayerRelaxRate = float(np.max(r,0)) #for single layer use r=0
#     infomapSimple.run()
#
#     rev_id_dict = create_tuple_indices(n, graph.nlayers, rev=True)
#     outmodules = np.array([-1 for _ in range(graph.N)])
#     cnt=0
#     for node in infomapSimple.iterTree():
#         if node.isLeaf():
#             cnt+=1
#             ind = rev_id_dict[(node.physicalId, node.layerId)]
#             outmodules[ind] = node.moduleIndex()
#
#     #some how you can have missing nodes in the output.  I believe these are dangling nodes but haven't checked
#     #we just leave this as -1 in the output community
#     # assert np.sum(outmodules == -1) == 0, "node module missing:{:}".format(str(np.where(outmodules == -1)[0]))
#     return outmodules

#python run_multilayer_matlab_test.py

def run_infomap_on_multiplex(n, nlayers, mu, p_eta, r, ntrials):
    ncoms=10
    finoutdir = os.path.join(matlabbench_dir, 'infomap_multiplex_matlab_test_data_n{:d}_nlayers{:d}_trials{:d}_{:d}ncoms_multilayer'.format(n,nlayers,ntrials,ncoms))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)

    output = pd.DataFrame()
    outfile="{:}/multiplex_test_n{:d}_L{:d}_mu{:.4f}_p{:.4f}_relax{:.4f}_trials{:d}.csv".format(finoutdir,n,nlayers,mu,p_eta,r,ntrials)

    qmax=10
    max_iters=4000
    print('running {:d} trials at r={:.3f}, p={:.4f}, and mu={:.4f}'.format(ntrials,r,p_eta,mu))
    for trial in range(ntrials):

        t=time()
        graph=create_multiplex_graph(n_nodes=n, mu=mu, p=p_eta,
                                     n_layers=nlayers, maxcoms=ncoms)
        print('time creating graph: {:.3f}'.format(time()-t))

        cind=output.shape[0]
        outpart=call_infomap(graph=graph, r=r)
        ami_layer=graph.get_AMI_layer_avg_with_communities(outpart)
        ami=graph.get_AMI_with_communities(outpart)
        output.loc[cind,'trial']=trial
        output.loc[cind,'mu']=mu
        output.loc[cind,'p']=p_eta
        output.loc[cind,'r']=r

        output.loc[cind, 'AMI'] = ami
        output.loc[cind, 'AMI_layer_avg'] = ami_layer


        if trial == 0:  # write out whole thing
            with open(outfile, 'w') as fh:
                output.to_csv(fh, header=True)
        else:
            with open(outfile, 'a') as fh:  # writeout last 2 rows for genlouvain + multimodbp
                output.iloc[-1:, :].to_csv(fh, header=False)

    return 0

def main():
    n = int(sys.argv[1]) #node in each layer i think
    nlayers=int(sys.argv[2])
    mu = float(sys.argv[3])
    p_eta= float(sys.argv[4])
    r=float(sys.argv[5])
    ntrials= int(sys.argv[6])
    #run_infomap_on_multiplex(n=200,nlayers=5,mu=0,p_eta=1.0,r=.1,ntrials=1)
    run_infomap_on_multiplex(n=n,nlayers=nlayers,mu=mu,p_eta=p_eta,r=r,ntrials=ntrials)

if __name__ == "__main__":
    #create_lfr_graph(n=1000, ep=.1, c=4, mk=12, use_gcc=True,orig=2,layers=2, multiplex = True)
    sys.exit(main())
