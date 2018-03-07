import numpy as np
import igraph as ig
import sklearn.metrics as skm

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
        print(block_sizes)
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