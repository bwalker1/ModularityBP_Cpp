import numpy as np
import random
import igraph as ig
from future.utils import iteritems,iterkeys
from collections import Hashable
import sklearn.metrics as skm
from .bp import BP_Modularity,PairVector
import itertools


class ModularityBP():
    """
    This is python interface class for the single layer modularity BP
    """

    def __init__(self,graph):
        self.graph=graph
        self.n=self.graph.vcount()
        self.m=self.graph.ecount()
        self.retrival_modularities={}
        self.marginals={} # should we keep these?
        self.partitions={} # max of marginals
        self.niters={}
        self.degrees=np.array(self.graph.degree())

        self.edgelist = self._get_edgelist()
        self._edgelistpv= self._get_edgelistpv()
        self._bpmod=None

    def run_modbp(self,beta,q,niter=100):

        #these doesn't appear to be working.  I don't think the mariginals
        #are resetting correclty for differnt betas .
        if self._bpmod is None:
            pv=PairVector(self.edgelist)
            self._bpmod=BP_Modularity(pv, _n=self.n, q=q, beta=beta,transform=False)
            iters=self._bpmod.run(niter)
        else:
            self._bpmod.setBeta(beta)
            self._bpmod.setq(q)
            iters=self._bpmod.run(niter)

        # self._bpmod = BP_Modularity(self._edgelistpv, _n=self.n, q=q, beta=beta, transform=False)
        # iters = self._bpmod.run(niter)
        cmargs=np.array(self._bpmod.return_marginals())
        cpartition = np.argmax(cmargs, axis=1)

        #assure it is initialized
        self.marginals[q]=self.marginals.get(q,{})
        self.partitions[q]=self.partitions.get(q,{})
        self.niters[q]=self.niters.get(q,{})
        self.retrival_modularities[q]=self.retrival_modularities.get(q,{})

        #set values
        self.niters[q][beta]=iters
        self.marginals[q][beta]=cmargs
        self.partitions[q][beta]=cpartition

        retmod=self._get_retrival_modularity(beta,q)
        self.retrival_modularities[q][beta]=retmod

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

    def _get_retrival_modularity(self,beta,q):
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
                cAmat = self.graph.get_adjacency()[cind, cind]

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
        return (1.0/(2.0*self.m))*(Ahat-(Phat/(2.0*self.m)))




