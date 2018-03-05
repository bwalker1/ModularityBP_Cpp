import numpy as np
import igraph as ig


class RandomERGraph():

    def __init__(self,n,p):
        self.graph=ig.Graph.Erdos_Renyi(n=n,p=p,directed=False,loops=False)

    def get_adjacency(self):
        return self.graph.get_adjacency()

    def get_edgelist(self):
        return self.graph.get_edgelist()


