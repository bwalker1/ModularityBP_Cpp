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

    def __init__(self,mlgraph=None,interlayer_edgelist=None,intralayer_edgelist=None,layer_vec=None):

        assert not (mlgraph is None) or not ( intralayer_edgelist is None and layer_vec is None)

        if mlgraph is not None:
            self.graph=mlgraph
        else:
            if interlayer_edgelist is None:
                interlayer_edgelist=np.array([])
            self.graph = MultilayerGraph(interlayer_edgelist,interlayer_edgelist,layer_vec)

        self.n=self.graph.n
        self.nlayers=self.graph.nlayers
        self.totaledgeweight=self.graph.totaledgeweight
        self.intralayer_edges=self.graph.intralayer_edges
        self.interlayer_edges=self.graph.interlayer_edges
        self.layer_vec=self.graph.layer_vec
        self._layer_vec_ia=IntArray(self.layer_vec)

        self.marginals={} # should we keep these?
        self.partitions={} # max of marginals
        self.niters={}


        rm_index=pd.MultiIndex(labels=[[],[],[],[]],levels=[[],[],[],[]],names=['q','beta','resgamma','omega'])
        self.retrival_modularities=pd.DataFrame(index=rm_index,columns=['retrival_modularity'])


        self._intraedgelistpv= self._get_edgelistpv()
        self._interedgelistpv= self._get_edgelistpv(inter=True)

        self._bpmod=None

    def run_modbp(self,beta,q,niter=100,resgamma=1.0,omega=1.0):

        if self._bpmod is None:
            self._bpmod=BP_Modularity(layer_membership=self._layer_vec_ia,
                                        intra_edgelist=self._intraedgelistpv,
                                      inter_edgelist=self._interedgelistpv,
                                      _n=self.n, _nt= self.nlayers , q=q, beta=beta,
                                      resgamma=resgamma,omega=omega,transform=False,verbose=True)

        else:
            if self._bpmod.getBeta() != beta:
                self._bpmod.setBeta(beta)
            if self._bpmod.getq() != q:
                self._bpmod.setq(q)
            if self._bpmod.getResgamma() != resgamma:
                self._bpmod.setResgamma(resgamma)
            if self._bpmod.getOmega() != omega:
                self._bpmod.setOmega(omega)

        iters=self._bpmod.run(niter)


        # self._bpmod = BP_Modularity(self._edgelistpv, _n=self.n, q=q, beta=beta, transform=False)
        # iters = self._bpmod.run(niter)
        cmargs=np.array(self._bpmod.return_marginals())

        # cpartition = np.argmax(cmargs, axis=1)
        cpartition=self._get_partition(cmargs)

        #assure it is initialized
        self.marginals[q]=self.marginals.get(q,{})
        self.partitions[q]=self.partitions.get(q,{})
        self.niters[q]=self.niters.get(q,{})
        # self.retrival_modularities[q]=self.retrival_modularities.get(q,{})

        #set values
        self.niters[q][beta]=iters
        self.marginals[q][beta]=cmargs
        self.partitions[q][beta]=cpartition

        retmod=self._get_retrival_modularity(beta,q,resgamma,omega)
        self.retrival_modularities.loc[(q,beta,resgamma,omega),'retrival_modularity']=retmod
        self.retrival_modularities.sort_index(inplace=True)

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
            return np.random.choice(np.flatnonzero(np.abs(x-x.max())<np.power(10.0,-4)))

        return np.apply_along_axis(func1d=argmax_breakties,arr=marginal,axis=1)

    def get_bstar(self,q):
        c=(2.0*self.totaledgeweight/(self.n))
        return np.log(q/(np.sqrt(c)-1)+1)

    def _get_retrival_modularity(self,beta,q,resgamma=1.0,omega=1.0):
        '''
        '''

        try:
            cpartition=self.partitions[q][beta]
        except KeyError:
            self.run_modbp(beta,q)
            cpartition=self.partitions[q][beta]

        #we sort indices into alike
        com_inddict = {}
        allcoms = sorted(list(set(cpartition)))
        sumA = 0

        Ahat = 0
        Chat = 0

        #For Ahat and Chat we simply iterate over the edges and count internal ones
        if self.intralayer_edges.shape[0]>0:
            Ahat=np.sum(np.apply_along_axis(func1d=lambda x: 1 if cpartition[x[0]]==cpartition[x[1]] else 0 , arr=self.intralayer_edges,axis=1))
        else:
            Ahat=0
        if self.interlayer_edges.shape[0]>0:
            Chat=np.sum(np.apply_along_axis(func1d=lambda x: 1 if cpartition[x[0]]==cpartition[x[1]] else 0 , arr=self.interlayer_edges,axis=1))
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




