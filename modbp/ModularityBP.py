import numpy as np
import random
import igraph as ig
from future.utils import iteritems,iterkeys
from collections import Hashable
import sklearn.metrics as skm
from .bp import BP_Modularity,PairVector
import itertools
import pandas as pd


class ModularityBP():
    """
    This is python interface class for the single layer modularity BP
    """

    def __init__(self,graph):
        self.graph=graph
        self.n=self.graph.vcount()
        self.m=self.graph.ecount()
        self.marginals={} # should we keep these?
        self.partitions={} # max of marginals
        self.niters={}
        self.degrees=np.array(self.graph.degree())

        rm_index=pd.MultiIndex(labels=[[],[],[]],levels=[[],[],[]],names=['q','beta','resgamma'])
        self.retrival_modularities=pd.DataFrame(index=rm_index,columns=['retrival_modularity'])


        self.edgelist = self._get_edgelist()
        self._edgelistpv= self._get_edgelistpv()
        self._bpmod=None

    def run_modbp(self,beta,q,niter=100,resgamma=1.0):
        #these doesn't appear to be working.  I don't think the mariginals
        #are resetting correclty for differnt betas .
        if self._bpmod is None:
            # pv=PairVector(self.edgelist)
            self._bpmod=BP_Modularity(self._edgelistpv, _n=self.n, q=q, beta=beta,
                                      resgamma=resgamma,transform=False)
            # print np.array(self._bpmod.return_marginals())
            # iters=self._bpmod.run(niter)
            # print np.array(self._bpmod.return_marginals())
        else:
            if self._bpmod.getBeta() != beta:
                self._bpmod.setBeta(beta)
            if self._bpmod.getq() != q:
                self._bpmod.setq(q)
            if self._bpmod.getResgamma() != resgamma:
                self._bpmod.setResgamma(resgamma)
            # print np.array(self._bpmod.return_marginals())
            # iters=self._bpmod.run(niter)
            # print np.array(self._bpmod.return_marginals())
        

        # # self._bpmod = BP_Modularity(self._edgelistpv, _n=self.n, q=q, beta=beta, transform=False)
        # # iters = self._bpmod.run(niter)
        # cmargs=np.array(self._bpmod.return_marginals())
        #
        # # cpartition = np.argmax(cmargs, axis=1)
        # cpartition=self._get_partition(cmargs)
        #
        # #assure it is initialized
        # self.marginals[q]=self.marginals.get(q,{})
        # self.partitions[q]=self.partitions.get(q,{})
        # self.niters[q]=self.niters.get(q,{})
        # # self.retrival_modularities[q]=self.retrival_modularities.get(q,{})
        #
        # #set values
        # self.niters[q][beta]=iters
        # self.marginals[q][beta]=cmargs
        # self.partitions[q][beta]=cpartition
        #
        # retmod=self._get_retrival_modularity(beta,q,resgamma)
        # self.retrival_modularities.loc[(q,beta,resgamma),'retrival_modularity']=retmod
        # self.retrival_modularities.sort_index(inplace=True)

    def _get_edgelist(self):
        edgelist=self.graph.get_edgelist()
        edgelist.sort()
        return edgelist

    def _get_edgelistpv(self):
        ''' Return PairVector swig wrapper version of edgelist'''
        if self.edgelist is None:
            self.edgelist=self._get_edgelist()
        _edgelistpv = PairVector(self.edgelist) #cpp wrapper for list
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
        c=(2.0*self.graph.ecount())/(self.graph.vcount())
        return np.log(q/(np.sqrt(c)-1)+1)

    def _get_retrival_modularity(self,beta,q,resgamma=1.0):
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

        # store indices for each community together in dict
        for i, val in enumerate(cpartition):
            try:
                com_inddict[val] = com_inddict.get(val, []) + [i]
            except TypeError:
                raise TypeError("Community labels must be hashable- isinstance(%s,Hashable): " % (str(val)), \
                                isinstance(val, Hashable))



        # convert indices to np_array
        for k, val in iteritems(com_inddict):
            com_inddict[k] = np.array(val)
        Phat=0
        Ahat=0
        for com in allcoms:
            cind = com_inddict[com]
            cdeg=self.degrees[cind]

            if cind.shape[0]==1:

                # there can be no edges within a one node community
                continue
                # cAmat = self.graph.get_adjacency()[cind, cind]
            else:
                #More efficiency way to do this?
                adj=np.array(self.graph.get_adjacency().data)
                cAmat=adj[np.ix_(cind, cind)]

                cPmat=np.outer(cdeg,cdeg.T)
                Phat+=np.sum(cPmat)
                Ahat+=np.sum(cAmat)
                # Phat+=np.sum(cPmat[np.tril_indices_from(cPmat,)])
                # Ahat+=np.sum(cAmat[np.tril_indices_from(cAmat,)])

            # if cind.shape[0] == 1:  # throws type error if try to index with scalar
            #     sumA += np.sum(adj_matrix[cind, cind])
            # else:
            #     sumA += np.sum(adj_matrix[np.ix_(cind, cind)])
        return (1.0/(2.0*self.m))*(Ahat-resgamma*(Phat/(2.0*self.m)))




