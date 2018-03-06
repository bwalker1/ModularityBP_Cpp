import numpy as np
import igraph as ig


class RandomERGraph():

    def __init__(self,n,p):
        self.graph=ig.Graph.Erdos_Renyi(n=n,p=p,directed=False,loops=False)

    def get_adjacency(self):
        return self.graph.get_adjacency()

    def get_edgelist(self):
        return self.graph.get_edgelist()


class RandomSBMGraph():

    def __init__(self,n,comm_prob_mat,block_sizes=None):
        if block_sizes is None:
            block_sizes=[n/(1.0*comm_prob_mat.shape[0]) for _ in range(comm_prob_mat.shape[0])]
        print(block_sizes)
        try:
            comm_prob_mat=comm_prob_mat.tolist()
        except TypeError:
            pass
        self.graph=ig.Graph.SBM(n=n,pref_matrix=comm_prob_mat,block_sizes=block_sizes,directed=False,loops=False)

    def get_adjacency(self):
        return self.graph.get_adjacency()

    def get_edgelist(self):
        return self.graph.get_edgelist()


#