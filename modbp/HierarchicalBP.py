import numpy
from .GenerateGraphs import MultilayerGraph
import sklearn.metrics as skm
from .bp import BP_Modularity,PairVector,IntArray,IntMatrix,DoubleArray


class HierarchicalBP():


    def __init__(self, mlgraph=None, interlayer_edgelist=None,
                 intralayer_edgelist=None, layer_vec=None,
                 accuracy_off=True, use_effective=False, comm_vec=None,
                 align_communities_across_layers_temporal=False,
                 align_communities_across_layers_multiplex=False,
                 normalize_edge_weights=False,
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

        assert not (mlgraph is None) or not (intralayer_edgelist is None and layer_vec is None)

        assert not (
                    align_communities_across_layers_multiplex and align_communities_across_layers_temporal), "Cannot use both multiplex and temporal alignment postprocessing.  Please set one to False"

        if mlgraph is not None:
            # this is just a single layer igraph. We create a mlgraph with empty interlayer edges
            if hasattr(mlgraph, 'get_edgelist'):
                self.graph = MultilayerGraph(intralayer_edges=np.array(mlgraph.get_edgelist()),
                                             interlayer_edges=np.zeros((0, 2), dtype='int'),
                                             layer_vec=[0 for _ in range(mlgraph.vcount())],
                                             is_bipartite=is_bipartite)

            else:
                self.graph = mlgraph

        else:
            if interlayer_edgelist is None:
                interlayer_edgelist = np.zeros((0, 2), dtype='int')
            self.graph = MultilayerGraph(intralayer_edges=intralayer_edgelist,
                                         interlayer_edges=interlayer_edgelist,
                                         layer_vec=layer_vec,
                                         is_bipartite=is_bipartite)

        if not comm_vec is None:
            self.graph.comm_vec = comm_vec


