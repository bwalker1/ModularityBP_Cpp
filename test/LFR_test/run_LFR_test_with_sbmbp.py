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
import sklearn.metrics as skm

#clusterdir="/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/test/LFR_test"
#arch = "elf64"

clusterdir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/test/LFR_test" #for testing locally
arch = "x86_64"

#clusterdir = "/Users/ben/Research (Github)/ModularityBP_Cpp/"
# finoutdir=os.path.join(clusterdir,'test/modbpdata/LFR_test_data_gamma3_beta2')



# orig_layers=[(1,100),(2,50),(3,33),(4,25),(5,20),(6,17),(10,10),(20,5),(50,2)]
orig_layers=[(1,40),(2,20),(4,10),(5,8),(10,4),(20,2),(40,1)]
#orig_layers=[(1,6),(2,3),(3,2),(6,1)] #for testing

def create_lfr_graph(n=1000, ep=.1, c=10, mk=20, use_gcc=True,orig=None,layers=None, multiplex = False):
    rprefix=np.random.randint(100000)
    rprefix_dir=os.path.join(clusterdir,str(rprefix))
    
    if multiplex:
        if orig is None or layers is None:
            raise ValueError("orig and layers must be set to generate multiplex network.")
        benchmarkfile = os.path.join(clusterdir,'MultiplexBenchmark/benchmark.'+arch)
        parameters = [
            benchmarkfile,
            "-N", '{:d}'.format(n),
            '-k', '{:.4f}'.format(c*layers),
            '-maxk', '{:d}'.format(mk*layers),
            '-mu', '{:.4f}'.format(ep),
            '-t1', '2', #gamma - exponent for degree sequence
            '-t2', '1', #beta - exponent for community size sequence
            '-minc', '100',
            '-maxc', '200',
            '-Orig','{:d}'.format(orig),
            '-L','{:d}'.format(layers)
        ]
    else:
        benchmarkfile = os.path.join(clusterdir,'binary_networks/benchmark.'+arch)
        parameters = [
            benchmarkfile,
            "-N", '{:d}'.format(n),
            '-k', '{:.4f}'.format(c),
            '-maxk', '{:d}'.format(mk),
            '-mu', '{:.4f}'.format(ep),
            '-t1', '2', #gamma
            '-t2', '1', #beta
            '-minc', '200',
            '-maxc', '300',
           '-w','{:d}'.format(rprefix_dir)
        ]
    os.system("mkdir {:}".format(rprefix_dir))
    process = Popen(parameters, stderr=PIPE, stdout=PIPE,
                    cwd = rprefix_dir)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError("creating LFR failed : {:}".format(stderr))
        
    # get the output network back into python
    if multiplex:
        tb = pd.read_table('{:}/level_node_node_weight.edges'.format(rprefix_dir),delimiter=' ',skiprows=1,header=None)
        m = tb[[0,1,2]].sort_values(by=[1,2]).as_matrix()
        #print m
        for i in xrange(len(m)): #this shift node ids from layer specific to global
            m[i,1] += n*(m[i,0]-1)
            m[i,2] += n*(m[i,0]-1)

        corig_offsets={}
        for layer in range(0,layers*orig):
            corig=layer//layers #which orig layer
            cv = pd.read_table('{:}/community_layer_{:d}'.format(rprefix_dir,layer),delimiter="\t",header=None)
            corig_offsets[corig]=len(np.unique(cv[1])) #number of communities
            coffset=np.sum([corig_offsets[i] for i in range(corig)])
            if layer==0:
                elist = m[:,[1,2]] - 1
                comvec = cv[1]
                layer_vec = np.zeros(n)
                inter_elist = np.zeros((0,2),dtype='int')
            else:
                elist = np.concatenate((elist,m[:,[1,2]] - 1))
                comvec = np.concatenate((comvec,cv[1]+coffset))
                layer_vec = np.concatenate((layer_vec,layer*np.ones(n)))
                new_elist = np.transpose((np.array(range(n))+(layer-1)*n,
                                          np.array(range(n))+layer*n))
                #print(new_elist)
                inter_elist = np.concatenate((inter_elist,new_elist))
    else:
         elist = pd.read_table('{:d}network.dat'.format(rprefix_dir), header=None).sort_values(
             by=0).as_matrix() - 1  # have to subtract 1 to get it to work
         # elist=elist[:elist.shape[0]/2,:]
         comvec = pd.read_table('{:d}community.dat'.format(rprefix_dir), header=None).sort_values(by=0).as_matrix()[:, 1] - 1
         layer_vec=[0 for _ in range(len(comvec))]
         inter_elist = None

    nfin=len(comvec)
    #print(inter_elist)
    mgraph = modbp.MultilayerGraph(intralayer_edges=elist, interlayer_edges=inter_elist, layer_vec=layer_vec,
                                   comm_vec=comvec)
    ##### plot community layout
    # plt.close()
    # f,a=plt.subplots(1,1)
    # mgraph.plot_communities(ax=a)
    # plt.show()
    # cgraph = mgraph.layers[0]
    # coms = np.unique(cgraph.vs['block'])
    # colors = sbn.color_palette('cubehelix', len(coms))
    # cdict = dict(zip(coms, colors))
    # cgraph.vs['color'] = map(lambda x: cdict[x], comvec)
    # if use_gcc:
    #     cgraph = cgraph.components().giant()
    for s in ['network.dat','community.dat','statistics.dat']:
        f2del="{:d}{:}".format(rprefix,s)
        #print (f2del)
        if os.path.exists(f2del):
            os.remove(f2del)
    if os.path.exists("{:}".format(rprefix_dir)):
        shutil.rmtree("{:}".format(rprefix_dir))
    return mgraph


# run SBMBP on the input graph with the chosen q, using the EM algorithm to learn parameters
# returns the AMI of the learned partition
def run_SBMBP_on_graph(graph):
    sbmbpfile = os.path.join(clusterdir,'test/mode_net/sbm')
    # outdir = os.path.join(clusterdir,'test/modbpdata/LFR_test_data/')
    rprefix = np.random.randint(100000)
    tmp_grph_file = os.path.join(finoutdir, '{:d}temporary_graph_file.gml'.format(rprefix))
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
        # print(stdout)
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
        if os.path.exists(marginal_file):
            os.remove(marginal_file)
    if os.path.exists(tmp_grph_file):
        os.remove(tmp_grph_file)


    minq = sorted(final_values.items(), key=lambda x: x[1]['f'])[0][0]

    AMI=skm.adjusted_mutual_info_score(all_partitions[q], graph.vs['block'])
    return AMI

#python run_LFR_test_with_sbmbp.py 100 .1 4 1.0 1 2 1.0
def main():
    n = int(sys.argv[1])
    ep = float(sys.argv[2])
    c = float(sys.argv[3])
    gamma = float(sys.argv[4])
    ntrials= int(sys.argv[5])
    try:
        orig,layers=orig_layers[int(sys.argv[6])]
        omega=float(sys.argv[7])
        ismultiplex=True

    except IndexError:
        ismultiplex=False

    finoutdir = os.path.join(clusterdir, 'LFR_test_data_n{:d}_trials{:d}_gamma2_beta1_k{:.2f}_multilayer'.format(n,ntrials,c))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)

    output=pd.DataFrame(columns=['ep','beta', 'resgamma', 'niters', 'AMI','retrieval_modularity','isSBM'])
    outfile="{:}/LFR_test_n{:d}_eps{:.4f}_gamma{:.4f}_omega{:.4f}_origlayers{:d}x{:d}_trials{:d}.csv".format(finoutdir,n, ep, gamma,omega,orig,layers,ntrials,)
    qmax=20
    max_iters=4000
    print('running {:d} trials at gamma={:.4f} and eps={:.4f}'.format(ntrials,gamma,ep))
    for trial in range(ntrials):

        graph=create_lfr_graph(n=n, ep=ep, c=c, mk=20, use_gcc=True,layers=layers,orig=orig,multiplex=True)
        graph.layers[0].save('test_LFR_onelayer.graphml.gz')
        # ami_sbm=run_SBMBP_on_graph(graph)
        # cind = output.shape[0]
        # output.loc[cind,['beta','resgamma','niters','retrieval_modularity']]=[None,None,None,None]
        # output.loc[cind,'AMI']=ami_sbm
        # output.loc[cind,'isSBM']=True
        # output.loc[cind,'ep']=ep
        mlbp = modbp.ModularityBP(mlgraph=graph,accuracy_off=True,use_effective=True,
                                  comm_vec=graph.comm_vec)
        bstars = [mlbp.get_bstar(q) for q in range(2, qmax)]
        #betas = np.linspace(bstars[0], bstars[-1], len(bstars) * 8)
        betas=bstars
        for beta in betas:
            mlbp.run_modbp(beta=beta, niter=max_iters, q=qmax, resgamma=gamma, omega=omega)

            mlbp_rm = mlbp.retrieval_modularities

        #print(mlbp_rm['niters']<1000 & mlbp_rm['is_trivial'] == False)
        #print (mlbp_rm['is_trivial'])
        ind2keep=np.where(np.logical_and(mlbp_rm['converged'],~mlbp_rm['is_trivial']))[0]
        cind = output.shape[0]
        if len(ind2keep)>0:
            minidx = mlbp_rm.iloc[ind2keep]['retrieval_modularity'].idxmax()
            for col in mlbp_rm.columns.values:
                output.loc[cind,col]=mlbp_rm.loc[minidx,col]
        else:
            for col in mlbp_rm.columns.values:
                output.loc[cind,col]=np.nan
            output.loc[cind,'converged']=False
            output.loc[cind,'niters']=max_iters+1

        output.loc[cind,'ep']=ep
        output.loc[cind,'orig']=orig
        output.loc[cind,'layers']=layers

        if trial == 0:
            with open(outfile, 'w') as fh:
                output.to_csv(fh, header=True)
        else:
            with open(outfile, 'a') as fh:  # writeout as we go
                output.iloc[[-1], :].to_csv(fh, header=False)

    return 0

if __name__ == "__main__":
    #create_lfr_graph(n=1000, ep=.1, c=4, mk=12, use_gcc=True,orig=2,layers=2, multiplex = True)
    main()
