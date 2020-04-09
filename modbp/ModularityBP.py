from __future__ import absolute_import
import numpy as np
import igraph as ig
from future.utils import iteritems,iterkeys
from collections import Hashable
from .GenerateGraphs import MultilayerGraph
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from .bp import BP_Modularity,PairVector,IntArray,IntMatrix,DoubleArray,DoublePairArray
import itertools
import pandas as pd
import scipy.optimize as sciopt
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sbn
from time import time
import warnings
import os,pickle,gzip
import logging
#logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=logging.ERROR)

class ModularityBP():

    """
    This is python interface class for the mulitlayer modularity BP.

    :cvar graph:
    :type graph:

    """

    def __init__(self, mlgraph=None, interlayer_edgelist=None,
                 intralayer_edgelist=None, layer_vec=None,
                 accuracy_off=True, use_effective=False, comm_vec=None,
                 align_communities_across_layers_temporal=False,
                 align_communities_across_layers_multiplex=False,
                 min_com_size=5, is_bipartite=False):

        """

        :param mlgraph:
        :param interlayer_edgelist:
        :param intralayer_edgelist:
        :param layer_vec:
        :param accuracy_off:
        :param use_effective:
        :param comm_vec:
        :param align_communities_across_layers_temporal:
        :param min_com_size:
        :param is_bipartite: if graph is bipartite, change underlying null model for intralayer \
        to use k_i d_j / m . note edges are still passed in as edge list and bipartiteness is not \
        checked for.
        """

        assert not (mlgraph is None) or not ( intralayer_edgelist is None and layer_vec is None)

        assert not (align_communities_across_layers_multiplex and align_communities_across_layers_temporal), "Cannot use both multiplex and temporal alignment postprocessing.  Please set one to False"

        if mlgraph is not None:
            # this is just a single layer igraph. We create a mlgraph with empty interlayer edges
            if hasattr(mlgraph, 'get_edgelist'):
                self.graph = MultilayerGraph (intralayer_edges=np.array(mlgraph.get_edgelist()),
                                              interlayer_edges=np.zeros((0,2),dtype='int'),
                                              layer_vec=[0 for _ in range(mlgraph.vcount())])

            else:
                self.graph=mlgraph

        else:
            if interlayer_edgelist is None:
                interlayer_edgelist=np.zeros((0,2),dtype='int')
            self.graph = MultilayerGraph(intralayer_edges=intralayer_edgelist,
                                         interlayer_edges=interlayer_edgelist,
                                         layer_vec=layer_vec,
                                         is_bipartite=is_bipartite)

        if not comm_vec is None:
            self.graph.comm_vec = comm_vec
        self.n=self.graph.N
        self.nlayers=self.graph.nlayers
        self.totaledgeweight=self.graph.totaledgeweight
        self.intralayer_edges=self.graph.intralayer_edges
        self.interlayer_edges=self.graph.interlayer_edges
        self._cpp_intra_weights=self._get_cpp_intra_weights()
        self._cpp_inter_weights=self._get_cpp_inter_weights()

        if hasattr(self.graph,"merged_layer"):
            self.layer_vec=self.graph.merged_layer
        else:
            ohe=skp.OneHotEncoder(categories='auto')
            self.layer_vec=np.array(ohe.fit_transform(self.graph.layer_vec.reshape(-1,1)).toarray())

        #
        self.layer_vec = [[int(i) for i in row] for row in self.layer_vec]
        self._layer_vec_ia=IntMatrix(self.layer_vec)
        self.layer_vec=np.array(self.graph.layer_vec)

        self.layers_unique=np.unique(self.layer_vec)
        self._accuracy_off=accuracy_off #calculating permuated accuracy can be expensive for large q
        self._align_communities_across_layers_temporal=align_communities_across_layers_temporal
        self._align_communities_across_layers_multiplex=align_communities_across_layers_multiplex

        self.marginals={}
        self.partitions={} # max of marginals
        self.niters={}
        self.marginal_index_to_close_marginals={} #maps each marginal index to common marginals (ie marginals that have the same information across nodes
        self.marginal_to_comm_number={} # the marginal index to the effective community number
        self.group_distances={}
        self.use_effective=use_effective
        self.nruns=0 #how many times has the BP algorithm been run.  Also serves as index for outputs
        self._permutation_vectors={} # This is used to keep track of which vertices are switched after as sweep.

        #make single index
        # rm_index=pd.MultiIndex(labels=[[],[],[],[]],levels=[[],[],[],[]],names=['q','beta','resgamma','omega'])
        self.retrieval_modularities=pd.DataFrame(columns=['q','beta','resgamma','omega',
                                                         'retrieval_modularity','niters'],dtype=float)

        self._intraedgelistpv= self._get_edgelistpv()
        self._interedgelistpv= self._get_edgelistpv(inter=True)

        self._bipart_class_ia =  self._get_bipart_vec()
        self.min_community_size = min_com_size  #for calculating true number of communities min number of node assigned to count.
        self._bpmod=None
        self._node2beliefsinds_dict=None

        if self.nlayers>1 and self.graph.is_bipartite:
            raise NotImplementedError("bipartite modularity belief propagation only available for single layer")

    def run_modbp(self, beta, q, niter=100, resgamma=1.0, omega=1.0, dumping_rate=1.0,
                  reset=True,
                  iterate_alignment=True,
                  niters_per_update=None,
                  starting_partition=None,
                  starting_marginals=None,
                  starting_SNR=10):
        """

        :param beta: The inverse tempature parameter at which to run the modularity belief propagation algorithm.  Must be specified each time BP is run.
        :param q:  The number of mariginals used for the run of modbp.  Note that if self.use_effective is true,  The final number of reported communities could be lower.
        :param niter:  Maximum number of iterations allowed.  If self._align_communities_across_layers is true, the actual number of runs could be higher than this upper bound though at most 2*niter
        :param resgamma:  The resolution parameter at which to run modbp.  Default is resgamma=1.0
        :param omega:  The coupling strength used in running multimodbp.  This represent how strongly the algorithm tries to assign nodes connected by an interlayer connection to the same community.
        :param reset:  If true, the marginals will be rerandomized when this method is called.  Otherwise the state will be maintained from previous runs if existing (assuming q hasn't changed).
        :return: None
        """
        tot_time=time()
        if beta==0:
            warnings.warn("beta cannot be zero.  Using 10-16")
            beta=np.power(10.0,-16)

        assert(q>0),"q must be > 0"
        q=int(q) #q must be int
        q_orig=q #before collapsing
        self.retrieval_modularities.loc[self.nruns, 'q'] = q_orig
        self.retrieval_modularities.loc[self.nruns, 'beta'] = beta
        self.retrieval_modularities.loc[self.nruns, 'omega'] = omega
        self.retrieval_modularities.loc[self.nruns, 'resgamma'] = resgamma
        if self.graph.is_bipartite:
            num_bipart = len (np.unique(self.graph.bipartite_classes))
        else:
            num_bipart = 1

        t=time()

        #if not supplied use the default when modbp object was created

        changes=[]
        if self._bpmod is None:
            self._bpmod=BP_Modularity(_n=self.n,
                                      layer_membership=self._layer_vec_ia,
                                      intra_edgelist=self._intraedgelistpv,
                                      intra_edgeweight=self._cpp_intra_weights,
                                      inter_edgelist=self._interedgelistpv,
                                      inter_edgeweight=self._cpp_inter_weights,
                                      _nlayers= self.nlayers , q=q, beta=beta,
                                      dumping_rate=dumping_rate,
                                      num_biparte_classes=num_bipart,bipartite_class=self._bipart_class_ia, #will be empty if not bipartite.  Found that had to make parameter mandatory for buidling swig Python Class
                                      resgamma=resgamma,omega=omega,transform=False,verbose=False,parallel=False)

        else:
            if self._bpmod.getBeta() != beta or reset:
                self._bpmod.setBeta(beta,reset=reset)
            if self._bpmod.getq() != q:
                self._bpmod.setq(q)
            if self._bpmod.getResgamma() != resgamma:
                self._bpmod.setResgamma(resgamma)
            if self._bpmod.getOmega() != omega:
                self._bpmod.setOmega(omega)
            if self._bpmod.getDumpingRate() != dumping_rate:
                self._bpmod.setDumpingRate(dumping_rate)

        assert (starting_marginals is None or starting_partition is None),'Cannot input both starting marginal and starting partition'

        if not starting_partition is None:
            start_margs=self.create_marginals_from_partition(starting_partition,SNR=starting_SNR)
            start_beliefs=self._create_beliefs_from_marginals(start_margs)
            self._set_beliefs(start_beliefs)

        if not starting_marginals is None:
            start_beliefs=self._create_beliefs_from_marginals(starting_marginals)
            self._set_beliefs(start_beliefs)

        if self._align_communities_across_layers_temporal or self._align_communities_across_layers_multiplex:
            iters_per_run=niter//2 #somewhat arbitrary divisor
        else:
            iters_per_run=niter # run as many times as possible on first run.
        logging.debug('creating modbp obj time: {:.4f}'.format(time()-t))
        t=time()
        #logging.debug('Running modbp at beta={:.3f}'.format(beta))
        converged=False

        if niters_per_update is None:
            niters_per_update=niter

        converged=False
        iters=0
        cnt=0
        itersper_dr=niters_per_update
        centrop=1.0
        while (not converged) and iters<niter:
            dr=dumping_rate
            changes=np.array(self._bpmod.run(itersper_dr))
            citers=len(changes)
            iters+=citers
            cnt+=1
            cmargs = np.array(self._bpmod.return_marginals())
            self.marginals[self.nruns] = cmargs
            centrop = _get_avg_entropy(cmargs)
            self._get_community_distances(self.nruns, use_effective=False)  # sets values in method
            cpartition = self._get_partition(self.nruns, use_effective=False)
            if self.graph.comm_vec is not None:
                cami = self.graph.get_AMI_layer_avg_with_communities(cpartition)
            else:
                cami = np.nan
            self.partitions[self.nruns] = cpartition
            _, cnts = np.unique(cpartition, return_counts=True)
            logging.debug('iters: {:d}, dr: {:.3f}, entropy : {:.4f}, AMI: {:.4f}, cnts:{:},last change {:.3e}'.format(iters,dr,centrop,cami,cnts,changes[-1]))
            if citers<itersper_dr:
                converged=True
                logging.debug('converged iters: {:d}, dr: {:.3f}, entropy : {:.3f}, AMI: {:.4f}, cnts:{:}, last change {:.3e}'.format(iters,dr,centrop,cami,cnts,changes[-1]))


        cmargs=np.array(self._bpmod.return_marginals())
        logging.debug('modbp run time: {:.4f}, {:d} iterations '.format(time() - t, iters))
        t=time()

        if iters<iters_per_run:
            converged=True
        self.marginals[self.nruns]=cmargs
        #Calculate effective group size and get partitions
        # logging.debug('Combining marginals')
        self._get_community_distances(self.nruns,use_effective=self.use_effective) #sets values in method
        cpartition = self._get_partition(self.nruns, self.use_effective)
        self.partitions[self.nruns] = cpartition

        if self.use_effective:
            q_new = self._merge_communities_bp(self.nruns)
            q = q_new
        # logging.debug('Combining mariginals time: {:.4f}'.format(time()-t))
        t=time()

        # if self._align_communities_across_layers and iters<niter:
        #     #logging.debug('aligning communities across layers')
        #     # print ("Bethe : {:.3f}, Modularity: {:.3f}".format(self._bpmod.compute_bethe_free_energy(),
        #     #                                                    self._get_retrieval_modularity(self.nruns)))
        #     nsweeps=self._perform_permuation_sweep(self.nruns) # modifies partition directly

        to_align = self._align_communities_across_layers_temporal or self._align_communities_across_layers_multiplex

        if to_align:
            alignment_function = self._perform_permuation_sweep_temporal if self._align_communities_across_layers_temporal else self._perform_permuation_sweep_multiplex


        if to_align and iters<niter:
            # logging.debug('aligning communities across layers')
            # print ("Bethe : {:.3f}, Modularity: {:.3f}".format(self._bpmod.compute_bethe_free_energy(),
            #                                                    self._get_retrieval_modularity(self.nruns)))
            t=time()
            nsweeps=alignment_function(self.nruns) # modifies partition directly
            if self.graph.comm_vec is not None:
                logging.debug("AMI after alignment: {:.3f}".format(self.graph.get_AMI_with_communities(self.partitions[self.nruns])))
            logging.debug('aligning communities across layers time: {:.4f} : nsweeps: {:d}'.format(time() - t,nsweeps))
            t = time()
            cnt=0


            # if self._align_communities_across_layers_multiplex :
            #     iterate_alignment=False

            #keep alternating with more BP runs and alignment sweeps until either
            #converged or we've exceded max number iterations
            while iterate_alignment and not (nsweeps==0 and converged==True) and not iters>niter:
                # plt.close()
                # f,a=plt.subplots(1,1,figsize=(6,6))
                # self.plot_communities(self.nruns,ax=a)
                # a.set_title('before rerunning')
                # plt.show()

                self._switch_beliefs_bp(self.nruns)
                #can't go more than the alloted number of runs
                changes = self._bpmod.run(iters_per_run)
                citers=len(changes)
                # plt.close()
                # f, a = plt.subplots(1, 1, figsize=(6, 6))
                # self.plot_communities(self.nruns, ax=a)
                # a.set_title('after rerunning')
                # plt.show()

                # logging.debug("BFE:{:.3f}".format(self._get_bethe_free_energy()))


                iters+=citers
                # print("cnt", cnt, 'iters', iters)
                if citers<iters_per_run: #it converged
                    converged=True

                cmargs = np.array(self._bpmod.return_marginals())
                self.marginals[self.nruns] = cmargs
                # Calculate effective group size and get partitions
                self._get_community_distances(self.nruns,self.use_effective)  # sets values in method
                cpartition = self._get_partition(self.nruns, self.use_effective)
                self.partitions[self.nruns] = cpartition
                if self.use_effective:
                    q_new = self._merge_communities_bp(self.nruns)
                    q = q_new
                cnt+=1

                logging.debug(
                    'after aligning time: {:.4f}, {:d} iterations more. total iters: {:d}.  Number align iteration:{:.3f}'.format(time() - t, citers,
                                                                                                                                  iters,cnt))
                t = time()
                # logging.debug("before persistence:{:.3f}".format(self._compute_persistence_multiplex(self.nruns)))
                nsweeps = alignment_function(self.nruns)  # modifies partition directly
                # logging.debug("after persistence:{:.3f}".format(self._compute_persistence_multiplex(self.nruns)))
            #final combined marginals

                logging.debug('aligning partitions and combing time: {:.4f}'.format(time() - t))
                logging.debug('nsweeps to permute: {:d}'.format(nsweeps))
                if converged and cnt>10: #only allow so many loops here
                    break #marginals are locked in at this point
        # Perform the merger on the BP side before getting final marginals
        if iters>=niter:
            logging.debug("Modularity BP did not converge after {:d} iterations.".format(iters))
            pass



        self.retrieval_modularities.loc[self.nruns, 'niters'] = iters
        if len(changes)>0:
            self.retrieval_modularities.loc[self.nruns, 'converged'] = changes[-1]<np.power(10.0,-8) #last change was small enough
        else:
            self.retrieval_modularities.loc[self.nruns, 'converged']=False
        retmod=self._get_retrieval_modularity(self.nruns)
        #logging.debug('calculating bethe_free energy')
        bethe_energy=self._get_bethe_free_energy()
        self.retrieval_modularities.loc[self.nruns,'retrieval_modularity']=retmod
        self.retrieval_modularities.loc[self.nruns,'bethe_free_energy']=bethe_energy
        self.retrieval_modularities.loc[self.nruns,'avg_entropy']=_get_avg_entropy(cmargs)
        _,com_cnts=np.unique(self.partitions[self.nruns],return_counts=True)

        # self.retrieval_modularities.loc[self.nruns,'num_coms']=self._get_true_number_of_communities(self.nruns)
        self.retrieval_modularities.loc[self.nruns, 'num_coms']=np.sum(com_cnts>=self.min_community_size)
       # self.retrieval_modularities.loc[self.nruns,'qstar']=self._get_true_number_of_communities(self.nruns)
        self.retrieval_modularities.loc[self.nruns,'bstar']=self.get_bstar(q,omega)
        if self.graph.comm_vec is not None:
            self.retrieval_modularities.loc[self.nruns,'AMI_layer_avg']=self.graph.get_AMI_layer_avg_with_communities(cpartition)
            self.retrieval_modularities.loc[self.nruns,'AMI']=self.graph.get_AMI_with_communities(cpartition)
            self.retrieval_modularities.loc[
                self.nruns, 'NMI_layer_avg'] = self.graph.get_AMI_layer_avg_with_communities(cpartition,useNMI=True)
            self.retrieval_modularities.loc[self.nruns, 'NMI'] = self.graph.get_AMI_with_communities(cpartition,useNMI=True)

            if not self._accuracy_off: #for low number of communities
                self.retrieval_modularities.loc[self.nruns,'Accuracy_layer_avg']=self.graph.get_accuracy_layer_averaged_with_communities(cpartition)

                self.retrieval_modularities.loc[self.nruns, 'Accuracy'] = self.graph.get_accuracy_with_communities(cpartition)

        self.retrieval_modularities.loc[self.nruns,'is_trivial']=self._is_trivial(self.nruns)

        logging.debug("Total modbp runtime : {:.3f}".format(time()-tot_time))
        self.nruns+=1


    def _get_bethe_free_energy(self):
        if self._bpmod is None:
            raise AssertionError( "cannot calculate the bethe free energy without running first.  Please call run_mobp.")
        return self._bpmod.compute_bethe_free_energy()



    def _get_cpp_intra_weights(self):
        #supply weights if none
        if self.graph.intralayer_weights is None:
            weights=[1.0 for i in range(len(self.graph.intralayer_edges)) ]
        else:
            weights=self.graph.intralayer_weights
        assert len(self.graph.intralayer_weights)==len(self.graph.intralayer_edges),"length of weights must match number of edges"
        layers=[]
        if hasattr(self.graph,'intralayer_layers') :
            #we are using the MergedMultilayerGraph with edges (i,j, layer , weight)
            layers=self.graph.intralayer_layers
        else:
            for e in self.graph.intralayer_edges:
                clayer=self.graph.layer_vec[e[1]]
                layers.append(float(clayer))

        layer_weights=np.array(list(zip(layers,weights)))
        return DoublePairArray(layer_weights)

    def _get_cpp_inter_weights(self):
        if self.graph.interlayer_weights is None:
            weights=[1.0 for i in range(len(self.graph.interlayer_edges)) ]
        else:
            weights=self.graph.interlayer_weights

        return DoubleArray(weights)


    def _get_edgelistpv(self,inter=False):
        ''' Return PairVector swig wrapper version of edgelist'''
        if inter:
            try: #found that type numpy.int doesn't work
                _edgelistpv = PairVector(self.interlayer_edges) #cpp wrapper for list
            except:
                self.interlayer_edges=[ (int(e[0]),int(e[1])) for e in self.interlayer_edges]
                _edgelistpv = PairVector(self.interlayer_edges)
        else:
            try:
                _edgelistpv = PairVector(self.intralayer_edges)
            except:
                self.intralayer_edges = [(int(e[0]), int(e[1])) for e in self.intralayer_edges]
                _edgelistpv = PairVector(self.intralayer_edges)

        return _edgelistpv

    def _get_bipart_vec(self):
        if self.graph.bipartite_classes is not None:
            bipart_classpv = IntArray (self.graph.bipartite_classes)
        else:
            bipart_classpv = IntArray ([])
        return bipart_classpv

    def _get_partition(self,ind,use_effective=True):
        """ We want to have argmax with randomly broken ties.

        :param ind: index of the marginal to use
        :return:
        """
        #thanks to SO 42071597

        marginal=self.marginals[ind]

        def argmax_breakties(x):
            try:
                return np.random.choice(np.flatnonzero(np.abs(x-x.max())<np.power(10.0,-6)))
            except:
                print(x)
                raise ValueError

        parts=np.apply_along_axis(func1d=argmax_breakties,arr=marginal,axis=1)


        if use_effective: #map the marginals to very close ones.
            groupmap=self.marginal_index_to_close_marginals[ind]
            # We use the effective communities to map
            parts=np.array( [ self.marginal_to_comm_number[ind][x] for x in parts])
            return parts

        else:
            return parts

    def _get_excess_degree(self):
        """get excess degree.  Note that this is unweighted degree """
        intradegrees=self.graph.get_intralayer_degrees(weighted=False)
        if len(intradegrees.shape)>1:
            intradegrees=np.sum(intradegrees,axis=1)

        degrees = np.append(intradegrees,self.graph.get_interlayer_degrees())

        # degrees = self.graph.intradegrees + self.graph.interdegrees
        d_avg = np.mean(degrees)
        d2=np.mean(np.power(degrees,2.0))
        return d2/d_avg - 1

    def get_bstar(self,q,omega=0):
        "Implementation to calculate bstar from Chen Shi et al 2018 (Weighted community\
         detection and data clustering using message passing)"


        ind2keep=np.where(np.logical_not(self.graph.self_loops_intra))[0]
        weights=np.array(self.graph.intralayer_weights)[ind2keep]
        if self.graph.nlayers > 1:
            ind2keep_inter=np.where(np.logical_not(self.graph.self_loops_inter))[0]

            weights=np.append(weights,omega * np.array(self.graph.interlayer_weights)[ind2keep_inter])

        def avg_weights(bstar, weights, q, c):
            # bstar should be scalar
            exp_b_w = np.exp(bstar * weights)
            return np.mean(np.power((exp_b_w - 1) / (exp_b_w + q - 1), 2.0)) * c - 1



        deg_excess=self._get_excess_degree()
        # print("deg_excess = {:.3f}".format(deg_excess))
        # plt.close()
        # bs = np.linspace(-.1, 2, 200)
        # plt.plot(bs, np.array(list(map(lambda x: avg_weights(x, weights=weights, q=q, c=deg_excess), bs))))
        # plt.show()
        # sols = sciopt.fsolve(avg_weights, x0=.5, args=(weights, q, deg_excess ))
        sols,r = sciopt.bisect(avg_weights, a=0,b=100, args=(weights, q, deg_excess ),full_output=True)

        if not r.converged:
            warnings.warn("Unable to compute bstar for inputs. Check weights of graph")
        return sols

    def _get_qval(self, bstar, omega):
        "given a choice of bstar and omega, what value of q was given intially"
        ind2keep = np.where(np.logical_not(self.graph.self_loops_intra))[0]
        weights = np.array(self.graph.intralayer_weights)[ind2keep]
        if self.graph.nlayers > 1:
            ind2keep_inter = np.where(np.logical_not(self.graph.self_loops_inter))[0]

            weights = np.append(weights, omega * np.array(self.graph.interlayer_weights)[ind2keep_inter])


        def avg_weights(q, weights, bstar, c):
            # bstar should be scalar
            exp_b_w = np.exp(bstar * weights)
            return np.mean(np.power((exp_b_w - 1) / (exp_b_w + q - 1), 2.0)) * c - 1

        deg_excess = self._get_excess_degree()
        q = sciopt.fsolve(avg_weights, x0=.5, args=(weights, bstar, deg_excess))[0]
        return q

    # def get_bstar(self,q,omega=0):
    #     #c is supposed to be the average excess degree
    #     # degrees=self.graph.intradegrees + self.graph.interdegrees
    #     # d_avg=np.mean(degrees)
    #     # d2=np.mean(np.power(degrees,2.0))
    #     # c= d2/d_avg - 1
    #     #c=(2.0*self.totaledgeweight/(self.n))
    #     # return np.log(q/(np.sqrt(c)-1)+1)
    #     if self._bpmod is None:
    #         self._bpmod=BP_Modularity(layer_membership=self._layer_vec_ia,
    #                                     intra_edgelist=self._intraedgelistpv,
    #                                   intra_edgeweight=self._cpp_intra_weights,
    #                                   inter_edgelist=self._interedgelistpv,
    #                                   _n=self.n, _nt= self.nlayers , q=q, beta=1.0, #beta doesn't matter here
    #                                    omega=omega,transform=False)
    #
    #     return self._bpmod.compute_bstar(omega,int(q)) #q must be an int

    def _get_retrieval_modularity(self,nrun=None):
        """

        :param nrun:
        :return:
        """
        if nrun is None:
            nrun=self.nruns #get last one

        resgamma,omega=self.retrieval_modularities.loc[nrun,['resgamma','omega']]
        cpartition = self.partitions[nrun] #must have already been run
        return calc_modularity(self.graph,partition=cpartition,resgamma=resgamma,omega=omega)
        

    def _get_community_distances(self,ind,use_effective=True,thresh=None):
        """
        Here we calculate the average distance between the mariginals of each of the \
        communities as defined by:

        :math:`d_{l,k}=\\frac{1}{N}\\sum_{i}(\\psi_{i}^{l}-\\psi_{i}^{k})^2` \

        We also identify communities that are close enough to be considered a single community\
        i.e. their distance is below the threshhold

        :return:
        """

        try:
            cmarginal = self.marginals[ind]
        except KeyError:
            raise KeyError("Cannot find partition with index {}".format(ind))



        #get direcly from the mariginals
        q=cmarginal.shape[1]

        # average values get closer as the number of marginals increases
        # fitted 2nd degree polynomial up to q=20 for large graph and
        # take 1/10 of average distances (from initialized values)
        if thresh == None:
            coefs = [0.00344322, -0.18963259, -0.85389837]

            def polycurve(x, coefs):
                tot = 0
                coefs = np.flip(coefs)
                for i, c in enumerate(coefs):
                    tot += c * np.power(x, i)
                return tot

            thresh = .1 * np.power(10.0, polycurve(q, coefs))
            # thresh=np.power(10.0,-3)
        distmat=np.zeros((q,q))

        # everyone starts out in their own group initially.
        # We merge sets together every time pairwise distance is less.
        groups=dict(zip(range(q),[{i} for i in range(q)]))

        if use_effective:
            #if not use effective we leave each marginal mapped to itself.
            for k,l in itertools.combinations(range(q),2):

                dist_kl=np.mean(np.power(cmarginal[:,k]-cmarginal[:,l],2.0))

                distmat[k,l]=dist_kl
                distmat[l,k]=dist_kl

                if dist_kl <=thresh:
                    comb=groups[l].union(groups[k])
                    for val in comb: #have to update everyone's groups
                        groups[val]=comb


        self.marginal_index_to_close_marginals[ind]=groups

        commsets = list(set([frozenset(s) for s in groups.values()]))
        revmap={}
        for i,comset in enumerate(commsets):
            for val in comset:
                #we use the minimum community mapped so that must number of communities
                #will retain mapping.
                revmap[val]=np.min(list(comset))

        #remap to those in the range from 0 to len(commsets)
        available=set(np.arange(len(commsets))).difference(set(revmap.values()))
        valsremap={}
        #use this dict to ensure that all vals are remapped consistently

        for k,val in list(set(revmap.items())):
            if val >= len(commsets):
                if val not in valsremap:
                    valsremap[val]=available.pop()
                revmap[k]=valsremap[val]

        if np.max(list(revmap.values()))>=len(set(revmap.values())):
            raise AssertionError

        self.marginal_to_comm_number[ind] = revmap



    def _groupmap_to_permutation_vector(self,ind):
        """
        Create vector that denotes communities that should be collapsed . \
        Each element of the array is the new community label for that index . \
        I.e [ 0, 0 , 1 , 2] Denotes a collpase from 4 communities to 3 where \
        the 0th and 1st old communities are merged into a single set of marginals.
        :param ind:
        :return:
        """
        revgroupmap=self.marginal_to_comm_number[ind]

        outarray=np.arange(np.max(list(revgroupmap.keys()))+1)

        for k,val in revgroupmap.items():
            outarray[k]=val

        return outarray
        # return IntArray(outarray)



    def _get_true_number_of_communities(self,ind):
        """

        :param ind:
        :return:
        """

        if ind not in self.marginal_index_to_close_marginals.keys():
            self._get_community_distances(ind)
        groupmap=self.marginal_index_to_close_marginals[ind]

        #create set of sets and take len.  Frozenset is immutable
        #
        if self.min_community_size==0:
            return len(set([frozenset(s) for s in groupmap.values()]))
        else:
            return len(set([ frozenset(s) for s in groupmap.values() if len(s) >= self.min_community_size ]))

    def _is_trivial(self,ind,thresh=None):
        """
        We use the same metric to define marginals that represent the same partitions\
        used in _get_community_distances.

        :param ind: index of marginal to examine
        :return: true if partition is close enough to trival, false if it is sufficiently differet
        """

        cmarginal=self.marginals[ind]
        q=cmarginal.shape[1]
        # mean distances were fit using numpy.polyfit based on number of marginals
        #for large network.
        if thresh == None:
            coefs = [0.00341228, -0.18898041, -0.85312958]

            def polycurve(x, coefs):
                tot = 0
                coefs = np.flip(coefs)
                for i, c in enumerate(coefs):
                    tot += c * np.power(x, i)
                return tot
            # thresh = np.power(10.0,-3)
            thresh = 2*np.power(10.0, polycurve(q, coefs))
        trival=np.ones(cmarginal.shape)/q
        if np.mean(np.power(cmarginal-trival,2.0))<=thresh:
            return True
        else:
            return False



    def _get_number_switched_single_layer(self,layer,partition,percent=False):

        layers = self.layers_unique

        if layer == 0:
            return 0
        else:
            # these are the identity connection across layers
            # we check for which identities switch
            prev_layer=layers[np.where(layers == layer)[0][0] - 1]
            interedges = self.graph.interedgesbylayers[(layer, prev_layer)] #use previous layer
            num_switched = 0
            for e in interedges:
                ei=e[0]
                ej=e[1]
                if partition[ei] != partition[ej]:
                    num_switched += 1

            if percent:
                num_switched /= float(len(interedges))
            return num_switched

    def _initialize_final_permutation_dict_all_layers(self,ind):
        """
        this object keeps track of which community every community is mapped to in each layer \
        in the permutation sweep to try and align the layers.  This is used at the end\
        to permute the marginals when re-running the modularity bp algorithm

        :param ind:
        :return: list with one dictionary for each layer mapping communities across layers
        """
        layers=self.layers_unique
        final_permutation_dict = [ ]
        for i,layer in enumerate(layers):
            #all the community labels across all layers
            partvals=np.unique(list(self.marginal_to_comm_number[ind].values()))
            #map to itself at begining
            final_permutation_dict.append(dict(zip(partvals,partvals)))
        return final_permutation_dict

    def _compute_persistence_multiplex(self,ind):

        """taken from https://epubs.siam.org/doi/pdf/10.1137/15M1009615.
        Persistence is number of communities that don't change when moving across a interlayeredge
        """
        curpart=self.partitions[ind]
        persistence=0
        for edge in self.interlayer_edges:
            n1,n2=edge
            if curpart[n1]==curpart[n2]:
                persistence+=1
        persistence=persistence/len(self.interlayer_edges)
        return persistence


    def _perform_permuation_sweep_multiplex(self, ind):
        """
        Calculate largest difference between adjacent layers\
        then perform flip for everylayer afterwards.  Repeat until \
        persistence no longer improves or max number of iterations are reached

        :param ind: partition to perform permutation on
        :return: number of sweeps performed. To keep track of whether \
        any layers were actually shuffled.
        """
        max_iters=100
        niters=-1 #first sweep doesn't count (if it only does one sweep then it didn't improve)

        #for each sweep perform we keep track of which communities are switched within each
        #layer.  We start out with each community mapped to itself.
        self._permutation_vectors[ind]=self._initialize_final_permutation_dict_all_layers(ind=ind)

        curpersistence=self._compute_persistence_multiplex(ind)
        prev_per=-np.inf
        distmat_dict = self._create_all_layer2layer_distmats(ind)
        while curpersistence-prev_per>0 and niters<max_iters:
            t=time()
            for layer in np.random.choice(self.layers_unique,replace=False,size=self.nlayers):

                #create permutation dictionary to the current layer based on best move
                #note that this diction
                permdict = self._create_layer_permutation_all_other_layer(ind,layer,distmat_dict)
                if np.all([ k==val for k,val in permdict.items() ]):
                    continue #nothing has changes
            #this dictionary of distances between layers gets updated
                distmat_dict=self._permute_layer_with_dict(ind,layer=layer,permutation=permdict,dismat_dict=distmat_dict)
                if(self._compute_persistence_multiplex(ind)-curpersistence)<0:
                    raise AssertionError
            #check that we are improving.
            prev_per=curpersistence
            curpersistence=self._compute_persistence_multiplex(ind)
            logging.debug('time performing 1 multiplex sweep: {:.3f}. Improvement: {:.3f}'.format(time()-t,curpersistence-prev_per))

            # print("Improvement:",curpersistence-prev_per,niters)
            niters+=1
        return niters


    def _perform_permuation_sweep_temporal(self, ind):
        """
        Calculate largest difference between adjacent layers\
        then perform flip for everylayer afterwards
        Repeat until no more flips are performed

        :param ind: partition to perform permutation on
        :return: number of sweeps performed. To keep track of whether \
        any layers were actually shuffled.
        """
        max_iters=100
        niters=0

        #for each sweep perform we keep track of which communities are switched within each
        #layer.  We start out with each community mapped to itself.
        self._permutation_vectors[ind]=self._initialize_final_permutation_dict_all_layers(ind=ind)

        while niters<max_iters: #this could also be a while loop but added max number of cycles
        # for clayer in self.layers_unique:
            #we swap the layer with the most number of mismatching nodes here
            number_switched = self.get_number_nodes_switched_all_layers(ind=ind, percent=True)
            max_layer_switched=np.argmax(number_switched)
            #create permutation dictionary to swap layer to nex
            #note that this diction
            permdict=self._create_layer_permutation_single_layer(ind,max_layer_switched)
            if all([k==v for k,v in permdict.items()]):
                break #nothing changed
            for layer in range(max_layer_switched,self.layers_unique[-1]+1): #permute all layers behind
                self._permute_layer_with_dict(ind,layer=layer,permutation=permdict)
            niters+=1
        return niters

    # def _perform_permuation_sweep_temporal(self, ind):
    #     """
    #     Calculate largest difference between adjacent layers\
    #     then perform flip for everylayer afterwards
    #     Repeat until no more flips are performed
    #
    #     :param ind: partition to perform permutation on
    #     :return: number of sweeps performed. To keep track of whether \
    #     any layers were actually shuffled.
    #     """
    #     max_iters=100
    #     niters=0
    #
    #     #for each sweep perform we keep track of which communities are switched within each
    #     #layer.  We start out with each community mapped to itself.
    #     self._permutation_vectors[ind]=self._initialize_final_permutation_dict_all_layers(ind=ind)
    #
    #
    #     while niters<max_iters: #this could also be a while loop but added max number of cycles
    #     # for clayer in self.layers_unique:
    #         #we swap the layer with the most number of mismatching nodes here
    #         number_switched = self.get_number_nodes_switched_all_layers(ind=ind, percent=True)
    #         max_layer_switched=np.argmax(number_switched)
    #         #create permutation dictionary to swap layer to nex
    #         #note that this diction
    #         permdict=self._create_layer_permutation_single_layer(ind,max_layer_switched)
    #
    #         if all([k==v for k,v in permdict.items()]):
    #             break #nothing changed
    #         for layer in range(max_layer_switched,self.layers_unique[-1]+1): #permute all layers behind
    #             self._permute_layer_with_dict(ind,layer=layer,permutation=permdict)
    #         niters+=1
    #     return niters


    def get_number_nodes_switched_all_layers(self, ind, percent=False):
        """
        For each layer, how many nodes switched from the previous layer ( 0 for first layer).\

        :param ind: index of the partitions to check for
        :param percent: = return percentage switched instead of number
        :return: array
        """

        cpart=self.partitions[ind]
        layers=self.layers_unique
        switched=np.zeros(len(layers))

        for i,layer in enumerate(layers):
            switched[i]=self._get_number_switched_single_layer(layer,cpart,percent=percent)

        return switched

    def _get_multiplex_layer_inds_dict(self,layer1,layer2):
        """
        returns 2 dictionaries, mapping the indices of the nodes in the previous layer to that \
        in the current layer. This is used to define the adjacent layer of nodes.

        creates interlayer_edge_dict={ node1 : [nodes with interlayer connections], node2 : []}
        if not already available.
        :return: layer1tolayer2_inds = dict mapping from current layer id to previous layer ids based on interalyer edges \
        layer2tolayer1_inds = dict mapping from previous layer to current layer.  Both these are based on the interlayer \
        edges and neither are strictly one-to-one


        """

        if "interlayer_edge_dict" not in self.__dict__:
            self.interlayer_edge_dict={}
            for e in self.interlayer_edges:
                ei,ej=e[0],e[1]
                self.interlayer_edge_dict[ei]=self.interlayer_edge_dict.get(ei,[])+[ej]
                self.interlayer_edge_dict[ej]=self.interlayer_edge_dict.get(ej,[])+[ei]

        layer1_inds=np.where(self.layer_vec==layer1)[0]
        layer2_inds=np.where(self.layer_vec==layer2)[0]
        layer2tolayer1_inds={}
        layer1tolayer2_inds={}
        for ind in layer2_inds:
            if ind in self.interlayer_edge_dict:
                layer2tolayer1_inds[ind]=list(set(self.interlayer_edge_dict[ind]).intersection(layer1_inds))

        #construct mapping other direction
        for prevind in layer2tolayer1_inds.keys():
            for cind in layer2tolayer1_inds.get(prevind,[]):
                layer1tolayer2_inds[cind]=list(set(layer1tolayer2_inds.get(cind,[])+[prevind]))

        return layer1tolayer2_inds,layer2tolayer1_inds


    def _get_previous_layer_inds_dict(self,layer):
        """
        returns 2 dictionaries, mapping the indices of the nodes in the previous layer to that \
        in the current layer. This is used to define the adjacent layer of nodes.

        creates interlayer_edge_dict={ node1 : [nodes with interlayer connections], node2 : []}

        :return: cur2prev_inds = dict mapping from current layer id to previous layer ids based on interalyer edges \
        prev2cur = dict mapping from previous layer to current layer.  Both these are based on the interlayer \
        edges and neither are strictly one-to-one


        """
        if layer==0: #nothing in the previous layer
            return {},{}

        if "interlayer_edge_dict" not in self.__dict__:
            self.interlayer_edge_dict={}
            for e in self.interlayer_edges:
                ei,ej=e[0],e[1]
                self.interlayer_edge_dict[ei]=self.interlayer_edge_dict.get(ei,[])+[ej]
                self.interlayer_edge_dict[ej]=self.interlayer_edge_dict.get(ej,[])+[ei]

        cur_inds=np.where(self.layer_vec==layer)[0]
        prev_inds=np.where(self.layer_vec==(layer-1))[0]
        prev2cur={}
        cur2prev={}
        for ind in prev_inds:
            if ind in self.interlayer_edge_dict:
                prev2cur[ind]=list(set(self.interlayer_edge_dict[ind]).intersection(cur_inds))

        #construct mapping other direction
        for prevind in prev2cur.keys():
            for cind in prev2cur.get(prevind,[]):
                cur2prev[cind]=list(set(cur2prev.get(cind,[])+[prevind]))

        return cur2prev,prev2cur



    # def _create_layer_permutation_single_layer(self,ind,layer):
    #     """
    #     Identify the permutation of community labels that minimizes the number\
    #     switched at the specified layer
    #
    #     :param ind:
    #     :return:
    #     """
    #
    #     cind = np.where(self.layer_vec == layer)[0]
    #     layers=self.layers_unique
    #     #we switch only the communiites in that layer
    #     layer_inds=np.where(self.layer_vec==layer)[0]
    #     prev_layer = layers[np.where(layers == layer)[0][0] - 1]
    #     prevind = np.where(self.layer_vec == prev_layer)[0]
    #     cur2prev_inds, prev2cur_inds = self._get_previous_layer_inds_dict(layer)
    #     prev_inds=list(prev2cur_inds.keys())
    #
    #     curpart = self.partitions[ind][cind]
    #     prevpart = self.partitions[ind][prevind]
    #     curcoms = np.unique(curpart)
    #     prevcoms = np.unique(prevpart)
    #     distmat = np.zeros((len(prevcoms), len(curcoms)))
    #
    #     # the index within the current layer partition
    #     prev_inds = {com: np.where(prevpart == com)[0] for com in prevcoms}
    #     cur_inds = {com: np.where(curpart == com)[0] for com in curcoms}
    #
    #     prevcoms2_i=dict(zip(prevcoms,range(len(prevcoms))))
    #     curcoms2_j=dict(zip(curcoms,range(len(curcoms))))
    #
    #     #this sets upf the distance matrix to compute optimal switches
    #     for prev_ind in prev2cur_inds.keys():
    #         pre_com=self.partitions[ind][prev_ind]
    #         i=prevcoms2_i[pre_com]
    #         for cur_ind in prev2cur_inds[prev_ind]:
    #             cur_com=self.partitions[ind][cur_ind]
    #             j=curcoms2_j[cur_com]
    #             #distmat[i, : ]+=(1.0/len(prev2cur_inds[prev_ind]))
    #             distmat[ i , : ]+=(1.0/len(prev2cur_inds[prev_ind]))
    #             distmat[ i , j ]-=(1.0/len(prev2cur_inds[prev_ind]))
    #     for cur_ind in cur2prev_inds.keys():
    #         cur_com = self.partitions[ind][cur_ind]
    #         j = curcoms2_j[cur_com]
    #         for prev_ind in cur2prev_inds[cur_ind]:
    #             prev_com = self.partitions[ind][prev_ind]
    #             i = prevcoms2_i[prev_com]
    #             # distmat[i, : ]+=(1.0/len(prev2cur_inds[prev_ind]))
    #             distmat[:, j] += (1.0 / len(cur2prev_inds[cur_ind]))
    #             distmat[i, j] -= (1.0 / len(cur2prev_inds[cur_ind]))
    #
    #
    #     #solve bipartite min cost matching with munkre algorithm
    #     row_ind,col_ind=sciopt.linear_sum_assignment(distmat)
    #     colcoms= list(map(lambda x : curcoms[x],col_ind))
    #     rwcoms= list(map(lambda x : prevcoms[x],row_ind))
    #     com_map_dict=dict(zip(colcoms,rwcoms)) #map to current layer coms to previous ones
    #
    #     #Mapping needs to be one-to-one so we have to fill in communities which weren't mapped
    #     # i.e communities that aren't in either of the layers
    #     coms_remaining=set(curcoms).difference(list(com_map_dict.values()))
    #     comsnotmapped=set(curcoms).difference(list(com_map_dict.keys()))
    #     #things that are in both get mapped to themselves first
    #     for com in coms_remaining.intersection(comsnotmapped):
    #         com_map_dict[com]=com
    #         coms_remaining.remove(com)
    #         comsnotmapped.remove(com)
    #     for com in comsnotmapped:
    #         com_map_dict[com]=coms_remaining.pop()
    #     return com_map_dict

    def _create_layer_permutation_single_layer(self,ind,layer):
        """
        Identify the permutation of community labels that minimizes the number\
        switched across all other layers (in multiplex context)

        :param ind:
        :return: com_map_dict mapping to apply to layer to minimize number of switches
        """

        #only do next layer

        #we switch only the communiites in that layer
        layer_inds=np.where(self.layer_vec==layer)[0]

        #set up distmat to include all possible communities
        # curcoms = np.unique(self.partitions[ind])
        curcoms=np.unique(list(self.marginal_to_comm_number[ind].values()))

        # we precompute these upfront for each sweep so we just have to combine
        distmat=self._create_layer_distmat(ind=ind,layer1=layer,layer2=layer-1)

        # distmat = np.zeros((len(curcoms), len(curcoms)))
        # for curlayer_compare in layers2compare:
        #     distmat+=distmat_dict[layer][curlayer_compare]

        #solve bipartite min cost matching with munkre algorithm
        row_ind,col_ind=sciopt.linear_sum_assignment(distmat)
        colcoms= list(map(lambda x : curcoms[x],col_ind))
        rwcoms= list(map(lambda x : curcoms[x],row_ind))
        com_map_dict=dict(zip(colcoms,rwcoms)) #map current layer coms to previous ones

        return com_map_dict

    def _create_layer_distmat(self,ind,layer1,layer2):

        # curcoms=np.unique(self.partitions[ind])
        curcoms=np.unique(list(self.marginal_to_comm_number[ind].values()))

        cur2other_inds, other2cur_inds = self._get_multiplex_layer_inds_dict(layer1, layer2)

        # the index within the current layer partition
        prevcoms2_i = dict(zip(curcoms, range(len(curcoms))))
        curcoms2_j = dict(zip(curcoms, range(len(curcoms))))
        distmat = np.zeros((len(curcoms), len(curcoms)))

        # this sets upf the distance matrix to compute optimal switches
        for prev_ind in other2cur_inds.keys():
            pre_com = self.partitions[ind][prev_ind]
            i = prevcoms2_i[pre_com]
            for cur_ind in other2cur_inds[prev_ind]:
                cur_com = self.partitions[ind][cur_ind]
                j = curcoms2_j[cur_com]
                distmat[i, :] += (1.0 / len(other2cur_inds[prev_ind]))
                distmat[i, j] -= (1.0 / len(other2cur_inds[prev_ind]))

        for cur_ind in cur2other_inds.keys():
            cur_com = self.partitions[ind][cur_ind]
            j = curcoms2_j[cur_com]
            for prev_ind in cur2other_inds[cur_ind]:
                prev_com = self.partitions[ind][prev_ind]
                i = prevcoms2_i[prev_com]
                # distmat[i, : ]+=(1.0/len(prev2cur_inds[prev_ind]))
                distmat[:, j] += (1.0 / len(cur2other_inds[cur_ind]))
                distmat[i, j] -= (1.0 / len(cur2other_inds[cur_ind]))

        return distmat

    def _create_all_layer2layer_distmats(self,ind):
        """

        :param ind:
        :return:
        """
        distmat_dict={}

        for i,layer in enumerate(self.layers_unique):
            if i==len(self.layers_unique):
                break
            for j,layer2 in enumerate(self.layers_unique[i+1:]):
                curdistmat=self._create_layer_distmat(ind,layer,layer2)
                distmat_dict[layer]=distmat_dict.get(layer,{})
                distmat_dict[layer2]=distmat_dict.get(layer2,{})
                distmat_dict[layer][layer2]=curdistmat
                distmat_dict[layer2][layer]=curdistmat.T #have distances both ways
        return distmat_dict

    def _create_layer_permutation_all_other_layer(self,ind,layer,distmat_dict):
        """
        Identify the permutation of community labels that minimizes the number\
        switched across all other layers (in multiplex context)

        :param ind:
        :return: com_map_dict mapping to apply to layer to minimize number of switches
        """

        cind = np.where(self.layer_vec == layer)[0]

        layers2compare=self.layers_unique
        layers2compare=layers2compare[np.where(layers2compare!=layer)[0]]


        #we switch only the communiites in that layer
        layer_inds=np.where(self.layer_vec==layer)[0]

        #set up distmat to include all possible communities
        # curcoms = np.unique(self.partitions[ind])
        curcoms=np.unique(list(self.marginal_to_comm_number[ind].values()))

        # we precompute these upfront for each sweep so we just have to combine
        distmat = np.zeros((len(curcoms), len(curcoms)))
        for curlayer_compare in layers2compare:
            distmat+=distmat_dict[layer][curlayer_compare]

        #solve bipartite min cost matching with munkre algorithm
        row_ind,col_ind=sciopt.linear_sum_assignment(distmat)
        colcoms= list(map(lambda x : curcoms[x],col_ind))
        rwcoms= list(map(lambda x : curcoms[x],row_ind))
        com_map_dict=dict(zip(colcoms,rwcoms)) #map current layer coms to previous ones

        return com_map_dict


    def _permute_layer_with_dict(self,ind,layer,permutation,dismat_dict=None):
        """

        Swap a given layer by the partition dictionary.  Any community \
        not present in dictionary is mapped to itself

        :param ind: which partition to permute
        :param layer: the layer that needs to be permuated in the
        :param permutation: dictionary mapping current values to new permuted community values
        :return: none

        """

        curcoms=np.unique(list(self.marginal_to_comm_number[ind].values()))
        lay_inds=np.where(self.layer_vec==layer)[0]
        old_layer=self.partitions[ind][lay_inds]
        #copy dictionary to add things later
        curpermutation=permutation.copy()

        #ensure that things don't get map to that aren't mapped so some other community
        comsremain=set(list(curcoms)+list(permutation.keys())).difference(permutation.values())
        coms2match=set(curcoms).difference(curpermutation.keys()) #communities that need to be matched

        for com in comsremain.intersection(coms2match) : # these map to themselve
            curpermutation[com]=com
            comsremain.remove(com)
            coms2match.remove(com)

        for com in coms2match:
            curpermutation[com]=comsremain.pop()



        self.partitions[ind][lay_inds]=\
            list(map(lambda x : curpermutation[x], self.partitions[ind][lay_inds]))

        #also apply map to the final permutation dictionary
        #No Communities should be merged or destroyed in this mapping
        #Each value (communities from previous permutation) should still be in the dictionary
        for k,val in self._permutation_vectors[ind][layer].items():
            if val in curpermutation: #otherwise not affected
                self._permutation_vectors[ind][layer][k]=curpermutation[val]


        assert len(set(self._permutation_vectors[ind][layer].values()))==len(self._permutation_vectors[ind][layer].values()), 'community lost in permutation'
        #sanity check.  Internal communities shouldn't change
        assert(np.abs(skm.adjusted_mutual_info_score(old_layer,self.partitions[ind][lay_inds],average_method='arithmetic')-1)<np.power(10.0,-6))

        #we can permute the columns of the distmats that have been affected
        #by the permutation

        if not dismat_dict is None:
            com2ind = dict(zip(curcoms,range(len(curcoms))))
            ind2com = dict(zip(range(len(curcoms)),curcoms))

            for k,distmat in dismat_dict[layer].items():
                #map from ind to community label then permuate then map back to ind
                #note all distmat have the same set of communities reprsented in
                #same order unless it's been permuted
                revorder={val:k for k,val in curpermutation.items()}
                perm_distmat=list(map( lambda x:  com2ind[revorder[ind2com[x]]]
                        ,range(len(curcoms))))
                #we only permute the distances from current to new, i.e. the columns
                newdist = distmat[:,perm_distmat]
                dismat_dict[layer][k]=newdist
                dismat_dict[k][layer]=newdist.T #have to switch out otherside as well.

            return dismat_dict


    def _create_all_layers_permuation_vector(self,ind):
        """
        We use the dictionary representation to create the permuation vector to pass into \
        the modularity bp\.  This is organized such that each row represents the current layer \
        And each position has the new position it is supposed to map to .  Ie [ 1 ,  ... ] \
        denotes that the zeroeth belief becomes the first belief (marginal) .
        :param ind:
        :return:
        """
        layers=self.layers_unique
        N=len(self.layers_unique)
        M=len(np.unique(list(self.marginal_to_comm_number[ind].values())))
        outarray=np.zeros((N,M)) #layers by #communites (after combining)

        curcoms=np.unique(list(self.marginal_to_comm_number[ind].values()))
        numcoms=len(set(self.marginal_to_comm_number[ind].values()))

        for i,layer in enumerate(layers):

            #use the final mapping dictionary to map each of the communities in this layer
            currow = list(zip(*sorted(list(self._permutation_vectors[ind][layer].items()),key=lambda x: x[1])))[0]
            # currow = list(map ( lambda  x : self._permutation_vectors[ind][layer][x],range(numcoms)))
            outarray[i,:]=currow

        return outarray

    def _merge_communities_bp(self,ind):
        """
        Merge communities that have the same marginals across all of the nodes
        :param ind:
        :return:
        """
        #type cast to int
        merge_vec=IntArray([int(x) for x in self._groupmap_to_permutation_vector(ind).astype(int)])
        # merge_vec=IntArray(self._groupmap_to_permutation_vector(ind).astype(int))
        self._bpmod.merge_communities(merge_vec)
        return len(set(self.marginal_to_comm_number[ind].values())) #new number of communities

    def _create_node_2_beliefs_dict(self,recreate=False,q=None):
        """For each node, get the indices of the incoming beliefs \
        so that we can pass a new belief into the _bpobj"""
        if q is None:
            assert not self._bpmod is None, "Must either specify q, or create the _bpmod obj"
            q=self._bpmod.getq()
        if self._node2beliefsinds_dict is None or recreate:
            node2beliefsinds_dict={}
            ecounts=np.array([ 0 for _ in range(self.graph.N)])
            #in the cpp beliefs are arrange in order of node indices
            #with all incoming beliefs contiguous ( and in blocks of q)
            for e in itertools.chain(self.graph.intralayer_edges, self.graph.interlayer_edges):
                if e[0]==e[1]:
                    continue
                ecounts[e[0]]+=1
                ecounts[e[1]]+=1
            ecounts=ecounts*q #factor in q
            cumsum_ecnt=np.cumsum(ecounts)
            for i,cnt in enumerate(cumsum_ecnt):
                if i==0:
                    node2beliefsinds_dict[i]=range(0,cnt)
                else:
                    node2beliefsinds_dict[i]=range(cumsum_ecnt[i-1],cnt)
            self._node2beliefsinds_dict=node2beliefsinds_dict

        return self._node2beliefsinds_dict

    def _get_belief_size(self,q=None):
        """calc size of belief vector"""
        node2beliefs=self._create_node_2_beliefs_dict(q=q)
        return np.sum([len(v) for v in node2beliefs.values()])

    def _create_beliefs_from_marginals(self,marginals):
        """We set all incoming beliefs to be the current marginal for the node"""
        q=marginals.shape[1]
        if not self._bpmod is None:
            assert self._bpmod.getq() == q , "Size of input marginals does not equal current beliefs"
        belief_size=self._get_belief_size(q=q)
        newbeliefs=np.array([-1.0 for _ in range(belief_size)])
        node2beliefinds=self._create_node_2_beliefs_dict(q=q)

        assert marginals.shape[1]==q ,"Marginals are not the correct shape"
        for i in range(marginals.shape[0]):
            cinds = node2beliefinds[i]
            #fill in new incoming beliefs for node i with copies of the marginal
            num2fill=len(cinds)//q
            newbeliefs[cinds]=np.array([ marginals[i,s] for s in range(q) for j in range(num2fill) ]).flatten()
        assert -1.0 not in newbeliefs, "one of new belief has not been created.  Check index dictionary"
        return newbeliefs

    def create_marginals_from_partition(self, partition, q=None, SNR=1000):

        if q is None:
            assert not self._bpmod is None, "Must either specify q, or create the _bpmod obj"
            q=self._bpmod.getq()


        outmargs = np.zeros((len(partition), q))
        for i in range(len(partition)):
            currow = np.array([1 for _ in range(q)])
            currow[int(partition[i])] = SNR
            currow = 1 / np.sum(currow) * currow
            outmargs[i, :] = currow
        return outmargs

    def _set_beliefs(self,beliefs):
        _da_beliefs=DoubleArray(beliefs)
        self._bpmod.setBeliefs(_da_beliefs)


    def _switch_beliefs_bp(self, ind):
        """
        This switches the belefs on the c++ side.  Should only be called after _perform_permutation_sweep
        :param ind: modbp run index
        :returns: permutes the beliefs or marginals
        """
        perm_vec_c=self._create_all_layers_permuation_vector(ind).astype(int)
        # type cast each to int
        perm_vec_c=IntMatrix([[int(x) for x in y] for y in perm_vec_c])

        self._bpmod.permute_beliefs(perm_vec_c)








    def plot_communities(self,ind=None,layers=None,ax=None,cmap=None):
        """

        :param ind:
        :param layers:
        :return:
        """


        if layers is None:
            layers=self.layers_unique


        def get_partition_matrix(partition, layer_vec):
            # assumes partiton in same ordering for each layer
            vals = np.unique(layer_vec)
            nodeperlayer = len(layer_vec) // len(vals)
            assert len(layer_vec)%len(vals) == 0 , "number of nodes in each layer must be equal"
            com_matrix = np.zeros((nodeperlayer, len(vals)))
            for i, val in enumerate(vals):
                cind = np.where(layer_vec == val)[0]
                ccoms = partition[cind]
                com_matrix[:, i] = ccoms
            return com_matrix


        cinds=np.where(np.isin(self.layer_vec,layers))[0]
        if ind is None: #use baseline
            assert self.graph.comm_vec is not None, "Must specify ground truth com_vec for graph"
            cpart=self.graph.comm_vec
        else:
            cpart=self.partitions[ind][cinds]

        vmin=np.min(cpart)
        vmax=np.max(cpart)

        clayer_vec=np.array(self.layer_vec)[cinds]
        part_mat=get_partition_matrix(cpart,clayer_vec)

        if ax is None:
            ax=plt.axes()

        if cmap is None:
            cmap=sbn.cubehelix_palette(as_cmap=True)


        ax.grid('off')
        ax.pcolormesh(part_mat,cmap=cmap,vmin=vmin,vmax=vmax)

        # numswitched=self.get_number_nodes_switched_all_layers(ind=ind,percent=True)
        # # numswitched=numswitched[np.where(np.isin(self.layers_unique,layers))[0]] #filter for layers selected
        # for i,num in enumerate(numswitched):
        #     ax.text(s="{:.2f}".format(num),x=i,y=-1,fontdict={"fontsize":9,'color':'white'})

        ax.set_xticks(range(0,len(layers)))
        ax.set_xticklabels(layers)
        return ax

def _get_avg_entropy(marginal):
    """caculate normalized entropies from marginals.  Ranges from 0 to 1"""
    entropies=[]
    for i in range(marginal.shape[0]):
        entropies.append(stats.entropy(marginal[i])/np.log(marginal.shape[1]))
    return np.mean(entropies)

def calc_modularity(graph,partition,resgamma,omega):
    """
    Calculate the modularity of graph for given partition, resolution, and omega
    :param graph: GenerateGraph.MultilayerGraph object
    :param partition:
    :param resgamma:
    :param omega:
    :return:
    """


    # we sort indices into alike
    com_inddict = {}
    allcoms = sorted(list(set(partition)))
    sumA = 0
    Ahat = 0
    Chat = 0

    def part_equal(x):
        if partition[int(x[0])] == partition[int(x[1])]:
            if len(x)>2:
                return x[2] #edge weights should be x[2]
            else:
                return 1.0
        else:
            return 0.0

    intra_edges=graph.intralayer_edges
    inter_edges = graph.interlayer_edges
    # zip weights into the edges
    if graph.intralayer_weights is not None:
        temp=[]
        for i,e in enumerate(intra_edges):
            temp.append((e[0],e[1],graph.intralayer_weights[i]))
        intra_edges=temp
    if graph.interlayer_weights is not None:
        temp=[]
        for i,e in enumerate(inter_edges):
            temp.append((e[0],e[1],graph.interlayer_weights[i]))
        inter_edges=temp

    # For Ahat and Chat we simply iterate over the edges and count internal ones
    if len(intra_edges) > 0:
        Ahat = np.sum(np.apply_along_axis(func1d=part_equal, arr=intra_edges, axis=1))
    else:
        Ahat = 0
    if not graph.is_directed:
        Ahat=Ahat*2.0

    if len(inter_edges) > 0:
        Chat = np.sum(np.apply_along_axis(func1d=part_equal, arr=inter_edges, axis=1))
    else:
        Chat = 0
    if not graph.is_directed:
        Chat = Chat * 2.0

    # We calculate Phat a little differently since it requires degrees of all members of each group
    # store indices for each community together in dict
    for i, val in enumerate(partition):
        try:
            com_inddict[val] = com_inddict.get(val, []) + [i]
        except TypeError:
            raise TypeError("Community labels must be hashable- isinstance(%s,Hashable): " % (str(val)), \
                            isinstance(val, Hashable))
    # convert indices stored together to np_array
    for k, val in iteritems(com_inddict):
        com_inddict[k] = np.array(val)

    Phat = 0
    degrees = graph.get_intralayer_degrees()  # get all layers degrees/strengths
    for i in range(graph.nlayers):
        layersum=0
        c_layer_inds = np.where(graph.layer_vec == i)[0]
        for com in allcoms:
            cind = com_inddict[com]
            # get only the inds in this layer
            cind_com = cind[np.where(np.isin(cind, c_layer_inds))[0]]
            cdeg = degrees[cind_com]  #
            if cind_com.shape[0] == 1:
                continue  # contribution is 0
            else:
                cPmat = np.outer(cdeg, cdeg.T)
                layersum += (np.sum(cPmat))
        layersum /= (2.0 * graph.intra_edge_counts[i])
        Phat += layersum

    # print("A:{:.2f},P:{:.2f},C:{:.2f}".format(Ahat,Phat,Chat))
    if graph.is_directed:
        mu = graph.totaledgeweight
    else:
        mu= 2.0*graph.totaledgeweight

    return (1.0 / mu) * (Ahat - resgamma * Phat + omega * Chat)