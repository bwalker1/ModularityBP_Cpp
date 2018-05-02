from __future__ import division
from context import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import sys
from subprocess import Popen,PIPE
import os
clusterdir="/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/"

def create_lfr_graph(n=1000, ep=.1, c=3, mk=10, use_gcc=True):
    benchmarkfile = os.path.join(clusterdir,'binary_networks/benchmark')
    print(benchmarkfile)
    rprefix=np.random.randint(100000)
    parameters = [
        benchmarkfile,
        "-N", '{:d}'.format(n),
        '-k', '{:.4f}'.format(c),
        '-maxk', '{:d}'.format(mk),
        '-mu', '{:.4f}'.format(ep),
        '-t1', '2',
        '-t2', '1',
        '-minc', '200',
        '-maxc', '300',
       '-w','{:d}'.format(rprefix)
    ]
    process = Popen(parameters, stderr=PIPE, stdout=PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError("creating LFR failed : {:}".format(stderr))

    elist = pd.read_table('{:d}network.dat'.format(rprefix), header=None).sort_values(
        by=0).as_matrix() - 1  # have to subtract 1 to get it to work
    # elist=elist[:elist.shape[0]/2,:]
    comvec = pd.read_table('{:d}community.dat'.format(rprefix), header=None).sort_values(by=0).as_matrix()[:, 1] - 1
    nfin=len(comvec)
    mgraph = modbp.MultilayerGraph(intralayer_edges=elist, layer_vec=[0 for _ in range(nfin)],
                                   comm_vec=comvec)
    cgraph = mgraph.layers[0]
    coms = np.unique(cgraph.vs['block'])
    colors = sbn.color_palette('cubehelix', len(coms))
    cdict = dict(zip(coms, colors))
    cgraph.vs['color'] = map(lambda x: cdict[x], comvec)
    if use_gcc:
        cgraph = cgraph.components().giant()
    for s in ['network.dat','community.dat','statistics.dat']:
        f2del="{:d}{:}".format(rprefix,s)
	#print (f2del)
	if os.path.exists(f2del):
            os.remove(f2del)
    return cgraph


# run SBMBP on the input graph with the chosen q, using the EM algorithm to learn parameters
# returns the AMI of the learned partition
def run_SBMBP_on_graph(graph):
    sbmbpfile = os.path.join(clusterdir,'test/mode_net/sbm')
    outdir = os.path.join(clusterdir,'test/modbpdata/LFR_test_data/')
    rprefix = np.random.randint(100000)
    tmp_grph_file = os.path.join(outdir, '{:d}temporary_graph_file.gml'.format(rprefix))
    graph.save(tmp_grph_file)
    all_partitions = {}
    final_values = {}
    for q in range(2, 5):
        parameters = [
            sbmbpfile, 'learn',
            "-l", tmp_grph_file,
            '-q', '{:d}'.format(q),
            '-M', '{:}_q{:d}_marginals.txt'.format(tmp_grph_file, q),
            '-d', '1',
            '-i', '1'
            #         '-L','{:}_q{:d}_planted_cab.txt'.format(grph_file,q),
            #         '--spcmode','{:d}'.format(0),
            #         '--wcab','{:}_q{:d}_cab.txt'.format(grph_file,q)
        ]
        process = Popen(parameters, stderr=PIPE, stdout=PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("running SBMBP failed : {:}".format(stderr))

        marginal_file = '{:}_q{:d}_marginals.txt'.format(tmp_grph_file, q)
        marginals = []
        partition = []
        inmargs = False
        inpartition = False
        with open(marginal_file, 'r') as f:

            for i, line in enumerate(f.readlines()):
                if re.search("\A\s*\Z", line):  # only while space
                    continue
                if i == 0:
                    fin_vals = dict([tuple(val.split('=')) for val in line.split()])
                    for k, val in fin_vals.items():
                        fin_vals[k] = float(val)
                    final_values[q] = fin_vals
                if re.search('marginals:', line):
                    inmargs = True
                    inpartition = False
                    continue
                if re.search('argmax_configuration', line):
                    inmargs = False
                    inpartition = True
                    continue
                if inmargs:
                    marginals.append(line.split())
                if inpartition:
                    partition = line.split()

        partition = np.array(partition, dtype=int)
        all_partitions[q] = partition
    minq = sorted(final_values.items(), key=lambda x: x[1]['f'])[0][0]
    return skm.adjusted_mutual_info_score(all_partitions[q], graph.vs['block'])



def main():
    n = int(sys.argv[1])
    q = int(sys.argv[2])
    ep = float(sys.argv[3])
    c = float(sys.argv[4])
    gamma = float(sys.argv[5])
    ntrials= int(sys.argv[6])
    output=pd.DataFrame(columns=['ep','beta', 'resgamma', 'niters', 'AMI','retrieval_modularity','isSBM'])
    outfile="{:}/test/modbpdata/LFR_test_data/LFR_test_n{:d}eps{:.4f}gamma{:.4f}trials{:d}.csv".format(clusterdir,n, ep, gamma,ntrials)
    print(outfile)
    qmax=8
    for trial in range(ntrials):

        graph=create_lfr_graph(n=n, ep=ep, c=c, mk=10, use_gcc=True)
        ami_sbm=run_SBMBP_on_graph(graph)
        cind = output.shape[0]
        output.loc[cind,['beta','resgamma','niters','retrieval_modularity']]=[None,None,None,None]
        output.loc[cind,'AMI']=ami_sbm
        output.loc[cind,'isSBM']=True
        mlbp = modbp.ModularityBP(mlgraph=graph,accuracy_off=True,use_effective=True,
                                  comm_vec=np.array(graph.vs['block']))
        bstars = [mlbp.get_bstar(q) for q in range(2, qmax)]
        betas = np.linspace(bstars[0], bstars[-1], len(bstars) * 4)
        for beta in betas:
            mlbp.run_modbp(beta=beta, niter=1000, q=qmax, resgamma=gamma, omega=0)

            mlbp_rm = mlbp.retrieval_modularities
        minidx = mlbp_rm[mlbp_rm['niters'] < 1000]['retrieval_modularity'].idxmax()
        cind=output.shape[0]
        output.loc[cind, ['beta', 'resgamma', 'niters', 'AMI','retrieval_modularity']] = mlbp_rm.loc[
            minidx, ['beta', 'resgamma', 'niters', 'AMI','retrieval_modularity']]
        output.loc[cind,['ep']]=ep

    output.to_csv(outfile)
    return 0

if __name__ == "__main__":
    main()
