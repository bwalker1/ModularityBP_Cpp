import numpy as np
import igraph as ig
from future.utils import iteritems,iterkeys
from collections import Hashable
import sklearn.metrics as skm
from .bp import BP_Modularity,PairVector
import itertools

class RandomGraph():
    def __init__(self):
        pass

    def get_adjacency(self):
        return self.graph.get_adjacency()

    def get_edgelist(self):
        return self.graph.get_edgelist()
    @property
    def m(self):
        return self.graph.ecount()

    @property
    def n(self):
        return self.graph.vcount()


class RandomERGraph(RandomGraph):

    def __init__(self,n,p):
        self.graph=ig.Graph.Erdos_Renyi(n=n,p=p,directed=False,loops=False)


class RandomSBMGraph(RandomGraph):
    def __init__(self,n,comm_prob_mat,block_sizes=None):
        if block_sizes is None:
            block_sizes=[int(n/(1.0*comm_prob_mat.shape[0])) for _ in range(comm_prob_mat.shape[0]-1)]
            block_sizes+=[n-np.sum(block_sizes)] #make sure it sums to one
        try:
            comm_prob_mat=comm_prob_mat.tolist()
        except TypeError:
            pass
        self.graph=ig.Graph.SBM(n=n,pref_matrix=comm_prob_mat,block_sizes=block_sizes,directed=False,loops=False)

        #nodes are assigned on bases of block
        block=[]
        for i,blk in enumerate(block_sizes):
            block+=[i for _ in range(blk)]
        self.graph.vs['block']=block

    @property
    def block(self):
        return self.graph.vs['block']

    def get_AMI_with_blocks(self,labels):
        """
        :param labels:
        :type labels:
        :return:
        :rtype:
        """
        return skm.adjusted_mutual_info_score(labels_pred=labels,labels_true=self.block)

class MultilayerSBM():

    def __init__(self):
        pass
        #TODO

    def get_interlayer_adj(self):
        #TODO
        return 0
    def get_intralayer_adj(self):
        #TODO
        return 0

    def get_layers(self,slice):
        #TODO
        return 0


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
        if self._bpmod is None:
            self._bpmod=BP_Modularity(self._edgelistpv,_n=self.n,q=q,beta=beta)
            self._bpmod.run(niter)
        else:
            self._bpmod.setBeta(beta)
            self._bpmod.setq(q)
            self._bpmod.run(niter)

        cmargs=np.array(self._bpmod.return_marginals())
        cpartition = np.argmax(cmargs, axis=1)
        #assure it is initialized
        self.marginals[q]=self.marginals.get(q,{})
        self.partitions[q]=self.partitions.get(q,{})
        self.retrival_modularities[q]=self.retrival_modularities.get(q,{})
        self.niters[q]=self.niters.get(q,{})
        #set values
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

        #first part is sum over edges so calculate from edge list
        def _mod(a):
            i = a[0]
            j = a[1]
            if cpartition[i] == cpartition[j]:
                # TODO make this weighted and add gamma
                return 1.0
            else:
                return 0.0
        Ahat=np.sum(np.apply_along_axis(func1d=_mod, arr=self.edgelist, axis=1))

        # convert indices to np_array
        for k, val in iteritems(com_inddict):
            com_inddict[k] = np.array(val)
        Phat=0
        for com in allcoms:
            cind = com_inddict[com]
            cdeg=self.degrees[cind]
            if cind.shape[0]==1:
                Phat+=cdeg
            else:
                cPmat=np.outer(cdeg,cdeg.T)
                #only sum lower triangular element of each
                Phat+=np.sum(cPmat[np.tril_indices_from(cPmat,-1)])
            # if cind.shape[0] == 1:  # throws type error if try to index with scalar
            #     sumA += np.sum(adj_matrix[cind, cind])
            # else:
            #     sumA += np.sum(adj_matrix[np.ix_(cind, cind)])
        return (1.0/self.m)*(Ahat-(1.0/(2*self.m))*Phat)




