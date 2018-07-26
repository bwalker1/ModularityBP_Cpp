from __future__ import division
# from context import modbp
import modbp
import numpy as np
import seaborn as sbn
import pandas as pd
import sys
from subprocess import Popen, PIPE
import re
import os
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import traceback
import scipy.sparse as scispa
import scipy.io as scio


#clusterdir = "/nas/longleaf/home/wweir/ModBP_proj/ModularityBP_Cpp/"
#clusterdir = "/nas02/home/w/w/wweir/ModBP_proj/ModularityBP_Cpp/"
clusterdir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/" #for testing locally

matlaboutdir = os.path.join(clusterdir,"test/genlouvain_mlsbm/matlab_transfer_file")
call_matlabfile = os.path.join(clusterdir,"test/genlouvain_mlsbm/call_matlab_genlouvain.sh")


# python run_sbm_ml_test.py 100 2 10 .1 5 .1 1 0.5 1.0
# python run_sbm_ml_test.py 250 2 20 0 10 0.2 1 2.0 .5
def main():

    # generate a graph and then run it some number of times
    n = int(sys.argv[1])
    q = int(sys.argv[2])
    nlayers = int(sys.argv[3])
    eta = float(sys.argv[4])
    c = float(sys.argv[5])
    ep = float(sys.argv[6])
    ntrials = int(sys.argv[7])
    omega = float(sys.argv[8])
    gamma = float(sys.argv[9])
    nblocks = q
    pin = c / (1.0 + ep * (q - 1.0)) / (n * 1.0 / q)
    pout = c / (1 + (q - 1.0) / ep) / (n * 1.0 / q)
    prob_mat = np.identity(nblocks) * pin + (np.ones((nblocks, nblocks)) - np.identity(nblocks)) * pout
    output = pd.DataFrame(columns=['ep', 'eta',  'resgamma', 'omega',
                                   'AMI', 'AMI_layer_avg', 'modularity'])


    finoutdir = os.path.join(clusterdir, 'test/genlouvain_mlsbm/sbm_test_data/SBM_test_data_n{:}_q{:d}_nt{:}'.format(n, q, ntrials))
    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)
    outfile = os.path.join(finoutdir,
                           "sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.2f}_gamma{:.2f}.csv".format(n, q, nlayers,
                                                                                                         eta, ep, omega,
                                                                                                         gamma))


    for trial in range(ntrials):
        ml_sbm = modbp.MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, transition_prob=eta)
        mgraph = modbp.MultilayerGraph(ml_sbm.intraedges, ml_sbm.layer_vec, ml_sbm.interedges,
                                       comm_vec=ml_sbm.get_all_layers_block())

        A, C = mgraph.to_scipy_csr()
        scio_outfile = os.path.join(matlaboutdir, 'sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.3f}_gamma{:.3f}_trial{:}.mat'.format(n,q,nlayers,eta,ep,omega,gamma,trial))
        matlaboutput = os.path.join(matlaboutdir, 'sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.3f}_gamma{:.3f}_trial{:}_output.mat'.format(n,q,nlayers,eta,ep,omega,gamma,trial))
        scio.savemat(scio_outfile, {"A": A, "C": C})
        parameters = [call_matlabfile,
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
        #print(stderr)
        #print(stdout)

        S = scio.loadmat(matlaboutput)['S'][:,0]
        ami=mgraph.get_AMI_with_communities(S)
        ami_layer_avg=mgraph.get_AMI_layer_avg_with_communities(S)
        mod=modbp.calc_modularity(mgraph,partition=S,resgamma=gamma,omega=omega)
        cind=output.shape[0]
        output.loc[cind, [ 'ep','eta','resgamma','omega','AMI', 'AMI_layer_avg','modularity']]= ep,eta,gamma,omega,ami,ami_layer_avg,mod
        # output.loc[cind, ['ep', 'eta']] = [ep, eta]
        if trial==0:
            with open(outfile,'w') as fh:
                output.to_csv(fh,header=True)
        elif trial>0:
            with open(outfile,'a') as fh: #writeout as we go
                output.iloc[[-1],:].to_csv(fh,header=False)
        try:
            os.remove(scio_outfile)
        except:
            print('could not remove {}'.format(scio_outfile))
        try:
            os.remove(matlaboutput)
        except:
            print('could not remove {}'.format(matlaboutput))

    return 0



def pydebug(type, value, tb):
    n = int(sys.argv[1])
    q = int(sys.argv[2])
    nlayers = int(sys.argv[3])
    eta = float(sys.argv[4])
    c = float(sys.argv[5])
    ep = float(sys.argv[6])
    ntrials = int(sys.argv[7])
    omega = float(sys.argv[8])
    gamma = float(sys.argv[9])


    finoutdir = os.path.join(clusterdir, 'test/modbpdata/SBM_test_data_n{:}_q{:d}_nt{:}/errors'.format(n, q, ntrials))

    if not os.path.exists(finoutdir):
        os.makedirs(finoutdir)
    erroroutfile = os.path.join(finoutdir,
                            "sbm_n{:d}_q{:d}_t{:d}_eta{:.2f}_ep{:.2f}_omega{:.2f}_gamma{:.2f}.error".format(n, q, nlayers,
                                                                                                          eta, ep,
                                                                                                          omega, gamma))
    with open(erroroutfile, 'w') as fh:
        #write error out to file
        # fh.write("Error type:" + str(type) + ": " + str(value))
        traceback.print_exception(type,value,tb,file=fh)
        # traceback.print_tb(tb,file=fh)

if __name__=='__main__':
        # sys.excepthook=pydebug
        sys.exit(main())
