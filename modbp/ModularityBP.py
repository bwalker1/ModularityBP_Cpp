import numpy as np
import random
import igraph as ig
from future.utils import iteritems,iterkeys
from collections import Hashable
from GenerateGraphs import MultilayerGraph
import sklearn.metrics as skm
from .bp import BP_Modularity,PairVector,IntArray
import itertools
import pandas as pd


class ModularityBP():
    """
    This is python interface class for the mulitlayer modularity BP
    """

    def __init__(self,mlgraph=None,interlayer_edgelist=None,intralayer_edgelist=None,layer_vec=None,accuracy_off=False):

        assert not (mlgraph is None) or not ( intralayer_edgelist is None and layer_vec is None)


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
                                         interlayer_edges=interlayer_edgelist,layer_vec=layer_vec)

        self.n=self.graph.n
        self.nlayers=self.graph.nlayers
        self.totaledgeweight=self.graph.totaledgeweight
        self.intralayer_edges=self.graph.intralayer_edges
        self.interlayer_edges=self.graph.interlayer_edges
        self.layer_vec=self.graph.layer_vec
        self._layer_vec_ia=IntArray(self.layer_vec)
        self.accuracy_off=accuracy_off #calculating permuated accuracy can be expensive for large q
        self.marginals={} # should we keep these?
        self.partitions={} # max of marginals
        self.niters={}
        self.group_maps={} #
        self.group_distances={}
        self.nruns=0 #how many times has the BP algorithm been run.  Also serves as index for outputs

        #make single index
        # rm_index=pd.MultiIndex(labels=[[],[],[],[]],levels=[[],[],[],[]],names=['q','beta','resgamma','omega'])
        self.retrieval_modularities=pd.DataFrame(columns=['q','beta','resgamma','omega',
                                                         'retrieval_modularity','niters'],dtype=float)


        self._intraedgelistpv= self._get_edgelistpv()
        self._interedgelistpv= self._get_edgelistpv(inter=True)

        self._bpmod=None

    def run_modbp(self,beta,q,niter=100,resgamma=1.0,omega=1.0,reset=False):

        if self._bpmod is None:
            self._bpmod=BP_Modularity(layer_membership=self._layer_vec_ia,
                                        intra_edgelist=self._intraedgelistpv,
                                      inter_edgelist=self._interedgelistpv,
                                      _n=self.n, _nt= self.nlayers , q=q, beta=beta,
                                      resgamma=resgamma,omega=omega,transform=False,verbose=False)

        else:
            if self._bpmod.getBeta() != beta or reset:
                self._bpmod.setBeta(beta)
            if self._bpmod.getq() != q:
                self._bpmod.setq(q)
            if self._bpmod.getResgamma() != resgamma:
                self._bpmod.setResgamma(resgamma)
            if self._bpmod.getOmega() != omega:
                self._bpmod.setOmega(omega)
                
        # in case c++ class calculated b*
        if beta==0:
            beta = self._bpmod.getBeta();


        iters=self._bpmod.run(niter)


        # self._bpmod = BP_Modularity(self._edgelistpv, _n=self.n, q=q, beta=beta, transform=False)
        # iters = self._bpmod.run(niter)
        cmargs=np.array(self._bpmod.return_marginals())

        # cpartition = np.argmax(cmargs, axis=1)
        cpartition=self._get_partition(cmargs)

        #assure it is initialized
        # self.marginals[q]=self.marginals.get(q,{})
        # self.partitions[q]=self.partitions.get(q,{})
        # self.niters[q]=self.niters.get(q,{})
        # self.retrieval_modularities[q]=self.retrieval_modularities.get(q,{})

        #set values
        self.marginals[self.nruns]=cmargs
        self.partitions[self.nruns]=cpartition
        self.retrieval_modularities.loc[self.nruns, 'q'] = q
        self.retrieval_modularities.loc[self.nruns, 'beta'] = beta
        self.retrieval_modularities.loc[self.nruns, 'niters'] = iters
        self.retrieval_modularities.loc[self.nruns, 'omega'] = omega
        self.retrieval_modularities.loc[self.nruns, 'resgamma'] = resgamma
        self._get_community_distances(self.nruns) #sets values in method

        retmod=self._get_retrieval_modularity(self.nruns)
        self.retrieval_modularities.loc[self.nruns,'retrieval_modularity']=retmod
        _,cnts=np.unique(cpartition,return_counts=True)

        self.retrieval_modularities.loc[self.nruns,'num_coms']=np.sum(cnts>5)
        self.retrieval_modularities.loc[self.nruns,'qstar']=self._get_true_number_of_communities(self.nruns)

        if self.graph.comm_vec is not None:
            self.retrieval_modularities.loc[self.nruns,'AMI_layer_avg']=self.graph.get_AMI_layer_avg_with_communities(cpartition)
            self.retrieval_modularities.loc[self.nruns,'AMI']=self.graph.get_AMI_with_communities(cpartition)

            if not self.accuracy_off:
                self.retrieval_modularities.loc[self.nruns,'Accuracy_layer_avg']=self.graph.get_accuracy_layer_averaged_with_communities(cpartition)

                self.retrieval_modularities.loc[self.nruns, 'Accuracy'] = self.graph.get_accuracy_with_communities(cpartition)


        # self.retrieval_modularities.loc[(q,beta,resgamma,omega),'retrieval_modularity']=retmod
        # self.retrieval_modularities.loc[(q,beta,resgamma,omega),'niters']=iters
        # self.retrieval_modularities.loc[(q,beta,resgamma,omega),'AMI']=self.graph.get_AMI_with_communities(cpartition)
        # self.retrieval_modularities.sort_index(inplace=True)

        self.nruns+=1

    def _get_edgelist(self):
        edgelist=self.graph.get_edgelist()
        edgelist.sort()
        return edgelist

    def _get_edgelistpv(self,inter=False):
        ''' Return PairVector swig wrapper version of edgelist'''
        if inter:
            _edgelistpv = PairVector(self.interlayer_edges) #cpp wrapper for list
        else:
            _edgelistpv = PairVector(self.intralayer_edges)
        return _edgelistpv

    def _get_partition(self,marginal):
        """ We want to have argmax with randomly broken ties.

        :param marginal:
        :return:
        """
        #thanks to SO 42071597

        def argmax_breakties(x):
            return np.random.choice(np.flatnonzero(np.abs(x-x.max())<np.power(10.0,-6)))

        return np.apply_along_axis(func1d=argmax_breakties,arr=marginal,axis=1)

    def get_bstar(self,q):
        #c is supposed to be the average excess degree
        degrees=self.graph.intradegrees + self.graph.interdegrees
        d_avg=np.mean(degrees)
        d2=np.mean(np.power(degrees,2.0))
        c= d2/d_avg - 1
        #c=(2.0*self.totaledgeweight/(self.n))
        return np.log(q/(np.sqrt(c)-1)+1)

    def _get_retrieval_modularity(self,nrun=None):
        '''
        '''
        if nrun is None:
            nrun=self.nruns #get last one

        resgamma,omega=self.retrieval_modularities.loc[nrun,['resgamma','omega']]
        cpartition = self.partitions[nrun] #must have already been run


        #we sort indices into alike
        com_inddict = {}
        allcoms = sorted(list(set(cpartition)))
        sumA = 0

        Ahat = 0
        Chat = 0

        def part_equal(x):
            if cpartition[x[0]]==cpartition[x[1]]:
                return 1
            else:
                return 0

        #For Ahat and Chat we simply iterate over the edges and count internal ones
        if self.intralayer_edges.shape[0]>0:
            Ahat=2*np.sum(np.apply_along_axis(func1d=part_equal, arr=self.intralayer_edges,axis=1))
        else:
            Ahat=0
        if self.interlayer_edges.shape[0]>0:
            Chat=2*np.sum(np.apply_along_axis(func1d=part_equal , arr=self.interlayer_edges,axis=1))
        else:
            Chat=0
        #TODO make this work for weighted edges

        # We calculate Phat a little differently since it requires degrees of all members of each group
        # store indices for each community together in dict
        for i, val in enumerate(cpartition):
            try:
                com_inddict[val] = com_inddict.get(val, []) + [i]
            except TypeError:
                raise TypeError("Community labels must be hashable- isinstance(%s,Hashable): " % (str(val)), \
                                isinstance(val, Hashable))
        # convert indices stored together to np_array
        for k, val in iteritems(com_inddict):
            com_inddict[k] = np.array(val)

        Phat=0
        degrees=self.graph.get_intralayer_degrees() #get all degrees
        for i in range(self.nlayers):
            c_layer_inds=np.where(self.graph.layer_vec==i)[0]
            #TODO for weighted network this should be the edge strengths
            for com in allcoms:
                cind = com_inddict[com]
                # get only the inds in this layer
                cind=cind[np.isin(cind,c_layer_inds)]
                cdeg=degrees[cind]#
                if cind.shape[0]==1:
                    continue #contribution is 0
                else:
                    cPmat=np.outer(cdeg,cdeg.T)
                    Phat+=(np.sum(cPmat)/(2.0*self.graph.intra_edge_counts[i]))


        return (1.0/(2.0*self.totaledgeweight))*( Ahat-resgamma*Phat+omega*Chat)

    def _get_community_distances(self,ind,thresh=np.power(10.0,-3)):
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

        q=cmarginal.shape[1]

        distmat=np.zeros((q,q))

        # everyone starts out in their own group initially.
        # We merge sets together every time pairwise distance is less.
        groups=dict(zip(range(q),[{i} for i in range(q)]))



        for k,l in itertools.combinations(range(q),2):
            dist_kl=np.mean(np.power(cmarginal[:,k]-cmarginal[:,l],2.0))

            distmat[k,l]=dist_kl
            distmat[l,k]=dist_kl

            if dist_kl <=thresh:
                comb=groups[l].union(groups[k])
                groups[k]=comb
                groups[l]=comb

        self.group_maps[ind]=groups
        self.group_distances[ind]=distmat

    def _get_true_number_of_communities(self,ind):
        """

        :param ind:
        :return:
        """

        if ind not in self.group_maps.keys():
            self._get_community_distances(ind)

        groupmap=self.group_maps[ind]

        #create set of sets and take len.  Frozenset is immutable
        return len(set([frozenset(s) for s in groupmap.values()]))



