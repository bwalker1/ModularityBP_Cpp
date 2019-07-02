import numpy as np
import sklearn.metrics as skm
import igraph as ig
import itertools  as it
import scipy.sparse as scispa
import matplotlib.pyplot as plt
import seaborn as sbn


"""Set of wrapper classes for easier access to graph information used by multimodbp"""

class RandomGraph(object):
    """
    Wrapper class for igraph with different accessor methods.
    """
    def __init__(self):
        pass

    def get_adjacency(self):
        return np.array(self.graph.get_adjacency().data)

    def get_edgelist(self):
        return self.graph.get_edgelist()
    @property
    def m(self):
        return self.graph.ecount()

    @property
    def n(self):
        return self.graph.vcount()



class RandomERGraph(RandomGraph):
    """
    Wrapper class to create igraph ER graph
    """

    def __init__(self,n,p):
        self.graph=ig.Graph.Erdos_Renyi(n=n,p=p,directed=False,loops=False)
        super(RandomERGraph,self).__init__()



class RandomSBMGraph(RandomGraph):
    """
    Wrapper class for a realization of non-degree corrected stochastic block model.
    """
    def __init__(self,n,comm_prob_mat,block_sizes=None,graph=None,use_gcc=False):
        """

        :param n: number of communities
        :param comm_prob_mat: Probabilities for node in community i to connect to node in community j
        :param block_sizes: How many nodes are in each community.  len(block_sizes) = num communities and \
        sum(block_sizes) = n
        :param graph: can create object from already existing igraph object or create from scratch given parameters.
        :param use_gcc:  if true, prune nodes node connected to largest component
        """
        if block_sizes is None:
            block_sizes = [int(n / (1.0 * comm_prob_mat.shape[0])) for _ in range(comm_prob_mat.shape[0] - 1)]
            block_sizes += [int(n - np.sum(block_sizes))]  # make sure it sums to one

        if graph is not None:
            self.graph=graph
        else:

            try:
                comm_prob_mat=comm_prob_mat.tolist()
            except TypeError:
                pass
            self.graph = ig.Graph.SBM(n=n, pref_matrix=comm_prob_mat, block_sizes=list(block_sizes), directed=False,loops=False)

        self.use_gcc=use_gcc
        self.block_sizes=np.array(block_sizes,dtype=int)
        self.comm_prob_mat=comm_prob_mat


        #nodes are assigned on bases of block
        block=[]
        for i,blk in enumerate(self.block_sizes):
            block+=[i for _ in range(blk)]
        self.graph.vs['block']=block

        if use_gcc==True: #in this case the block sizes need ot be recalculated
            self.graph=self.graph.components().giant()
            _,cnts=np.unique(self.graph.vs['block'],return_counts=True)
            self.block_sizes=cnts

    @property
    def block(self):
        '''vector denoting the membership of each node'''
        return self.graph.vs['block']

    def get_AMI_with_blocks(self,labels):
        """
        Compare partition (labels) of the nodes with the ground truth of the SBM

        :param labels: a labeling of the nodes
        :type labels: iterable
        :return: AMI
        :rtype: float
        """
        return skm.adjusted_mutual_info_score(labels_pred=labels,labels_true=self.block)

    def get_pin_pout_ratio(self):
        """

        Calculate :math: `\epsilon = \frac{p_{in}}{p_{out}}` for the SBM.

        :return:
        """
        totpin=0
        totpout=0
        for ei,ej in self.get_edgelist():
            if self.graph.vs['block'][ei]!=self.graph.vs['block'][ej]:
                totpout+=1
            else:
                totpin+=1

        #total number of out edges
        pin_possible=np.sum([ bs*(bs-1.0) for bs in self.block_sizes])
        pout_possible=self.n*(self.n-1.0)-pin_possible #all other possible edges have to be external

        return (totpin/pin_possible)/(totpout/pout_possible)

    def get_observed_group_sizes(self):
        """
        Get the block sizes of the underlying graph.  Especially useful when we have taken the GCC \
        so block sizes might have changed from specified parameters.
        :return:
        """
        coms,cnts=np.unique(self.graph.vs['block'],return_counts=True)
        return cnts

    def get_observed_cin_cout(self):
        """

        :return: degree for each of the blocks
        """
        coms,cnts=np.unique(self.graph.vs['block'],return_counts=True)

        ncoms=len(coms)

        totalcnts=np.divide(np.ones((ncoms, ncoms)) * sum(cnts),
                     np.outer(cnts,cnts)-np.diag(cnts)) #N/(N_A*N_B)
        label2num=dict(list(zip(coms,range(ncoms))))
        observed_cnts=np.zeros((ncoms,ncoms))

        for ei, ej in self.get_edgelist():
            ind1=label2num[self.graph.vs['block'][ei]]
            ind2=label2num[self.graph.vs['block'][ej]]
            observed_cnts[ind1,ind2]+=1
            observed_cnts[ind2,ind1]+=1
        return np.multiply(observed_cnts,totalcnts)



    def get_accuracy(self, labels):
        """
        :param labels: labels to compare to known blocks
        :type labels: list
        :return:
        :rtype:
        """
        return skm.accuracy_score(labels, self.block)

class MultilayerGraph(object):
    """
    Wrapper class for storing a 'multilayer graph'  that can be used to call the modularity belief propagation\
    A graph here is represented by a collection of igraphs (each one representing a "layer") as well as a set of \
    edges between the layers.  In this formulation, each node can only be present in a single layer
    """

    def __init__(self,intralayer_edges,layer_vec,interlayer_edges=None,comm_vec=None,
                 directed=False):
        """

        :param intralayer_edges: list of intralayer edges between the nodes. If intralayer_edges.shape[1] > 2\
         intralayer_edges[:,2] is assumed to represent the weights of the edges. Default weight is 1.
        :param layer_vec: vector denoting layer membership for each edge.  Size of network is taken to be\
        len(layer_vec)
        :param interlayer_edges: list of edges across layers.  If interlayer_edges.shape[1] > 2\
         interlayer_edges[:,2] is assumed to represent the weights of the edges. Default weight is 1.
        :param comm_vec: Underlying known communitiies of the network.  Default is None
        :param directed:  Are intralayer and interlayer edges directed.  #TODO: allow one or the other to be directed.
        """

        self.N=len(layer_vec)
        self.layer_vec=np.array([int(x) for x in layer_vec]) #NEEDS to be integer valued.
        self.intralayer_edges=intralayer_edges
        self.is_directed=directed
        self.unweighted=True



        #create an vector length zero
        if interlayer_edges is None: #Assume that it is single layer
            self.interlayer_edges=np.zeros((0,2),dtype='int')
            self.interlayer_weights=None
        else:
            self.interlayer_edges=interlayer_edges
            #are interlayer weights presen
            if len(self.interlayer_edges)>0 and len(self.interlayer_edges[0])>2:#weights are present
                self.interlayer_weights = [e[2] for e in self.interlayer_edges]
                self.interlayer_edges = [ (e[0],e[1]) for e in self.interlayer_edges]
                self.unweighted=False
            else:
                self.interlayer_weights=[ 1.0 for _ in range(len(self.interlayer_edges))]

        #are intralayer weights present
        if len(self.intralayer_edges)>0 and len(self.intralayer_edges[0]) > 2:  # weights are present
            self._intralayer_weights = [e[2] for e in self.intralayer_edges]
            self.intralayer_edges = [(e[0], e[1]) for e in self.intralayer_edges]
            self.unweighted=False
        else:
            self._intralayer_weights = [1.0 for _ in range(len(self.intralayer_edges))]

        if not self.is_directed:
            self._prune_intra_edges_for_undirected()  # make sure each edge is unique
            self._prune_inter_edges_for_undirected()

        self.layers=self._create_layer_graphs()
        self.nlayers=len(self.layers)
        self.intradegrees=self.get_intralayer_degrees() #by default these are weighted
        self.interdegrees=self.get_interlayer_degrees()
        self.intra_edge_counts=self.get_layer_edgecounts()
        if self.is_directed:
            self.totaledgeweight=np.sum(self.interdegrees)+np.sum(self.intradegrees)
        else:
            self.totaledgeweight=np.sum(self.interdegrees)/2.0+np.sum(self.intradegrees)/2.0

        self.comm_vec=comm_vec #for known community labels of nodes
        if self.comm_vec is not None:
            self._label_layers(self.comm_vec)
        self.interedgesbylayers=self._create_interlayeredges_by_layers()

    @property
    def intralayer_weights(self):
        return self._intralayer_weights

    @intralayer_weights.setter
    def intralayer_weights(self, intra_weights):
        """We have to recreate the layer graphs to be able to access degrees and\
        strengths if these have changed"""
        self._intralayer_weights = intra_weights
        self.layers=self._create_layer_graphs()


    def _prune_intra_edges_for_undirected(self):
        eset={}
        for i,e in enumerate(self.intralayer_edges): #note that we assume here BOTH WEIGHTs will be the same if edges are duplicated
            if e[0]<e[1]:
                eset[(e[0], e[1])] = self._intralayer_weights[i]
            else:
                eset[(e[1], e[0])] = self._intralayer_weights[i]
        edges=[]
        weights=[]
        for k,val in eset.items():
            edges.append(k)
            weights.append(val)

        edge_weights=sorted(list(zip(edges,weights)),key=lambda x:x[0])
        edges,weights=list(zip(*edge_weights))
        self.intralayer_edges=edges
        self.intralayer_weights=weights

    def _prune_inter_edges_for_undirected(self):
        #interlayer edges aren't weighted.  We just remove duplicated edges
        eset=set([])
        for i, e in enumerate(self.interlayer_edges):
            if e[0] < e[1]:
                eset.add((e[0], e[1]))
            else:
                eset.add((e[1], e[0]))
        eset=sorted(list(eset), key= lambda x:x[0])
        self.interlayer_edges=eset





    def _create_layer_graphs(self):
        layers=[]
        uniq=np.unique(self.layer_vec)
        for val in uniq:
            node_inds=np.where(self.layer_vec==val)[0]
            min_ind=np.min(node_inds)
            node_inds=set(node_inds) # hash for look up
            celist=[]
            cweights=[]
            #subtract this off so that number of nodes created in igraph is correct
            for i,e in enumerate(self.intralayer_edges):
                ei,ej=e[0],e[1]
                weight = 1.0 if self._intralayer_weights is None else self._intralayer_weights[i]
                if ei in node_inds or ej in node_inds:
                    if not (ei>=min_ind and ej>=min_ind):
                        raise AssertionError('edge indicies not in layer {:d},{:d}'.format(ei,ej))
                    celist.append((ei-min_ind,ej-min_ind))
                    cweights.append(weight)
            layers.append(self._create_graph_from_elist(len(node_inds),celist,cweights))
        return layers


    def _create_graph_from_elist(self,n,elist,weights=None,simplify=True):
        '''

        :param n: number of nodes
        :param elist: list of edges
        :param weights: vector of weights of the edges.  Len(weights) must equal len(elist).
        :param simplify:  remove multi-edges
        :return:
        '''
        cgraph=ig.Graph(n=n,edges=elist,directed=False)
        if weights is not None:
            cgraph.es['weight']=weights
        if simplify:
            cgraph=cgraph.simplify(multiple=True,combine_edges='sum')

        return cgraph

    def _create_interlayeredges_by_layers(self):
        layers2edges={}

        for e in self.interlayer_edges:
            i,j=e[0],e[1]
            lay_i=self.layer_vec[i]
            lay_j=self.layer_vec[j]
            layers2edges[(lay_i,lay_j)]=layers2edges.get((lay_i,lay_j),[])+[(i,j)]
            layers2edges[(lay_j, lay_i)]=layers2edges[(lay_i, lay_j)] #keep reference
        return layers2edges

    def _label_layers(self,comvec=None):
        """
        Here we set the true community assignment for each node in each layer using commvec
        :return: None
        """
        if comvec is None:
            assert self.comm_vec is not None, "Cannot set node communities if they are not provided"
            comvec=self.comm_vec

        assert len(comvec)==self.N,"length of comvec: {:d} does not equal number of nodes: {:}".format(len(comvec),self.N)
        coffset=0 #keep track of nodes already seen
        for layer in self.layers:
            layer.vs['block']=comvec[coffset:coffset+layer.vcount()]
            coffset+=layer.vcount()



    def get_layer_edgecounts(self):
        """m for each layer"""
        ecounts=[]
        for i in range(self.nlayers):
            if self.is_directed:
                ecounts.append(np.sum(self.get_intralayer_degrees(i)))
            else:
                ecounts.append(np.sum(self.get_intralayer_degrees(i))/2.0)

        return np.array(ecounts)

    def get_intralayer_degrees(self, i=None,weighted=True):
        if i is not None:
            if weighted and 'weight' in self.layers[i].es.attributes():
                return np.array(self.layers[i].strength(weights='weight'))
            else:
                return np.array(self.layers[i].degree())
        else:
            total_degrees=[]
            for i in range(len(self.layers)):
                if weighted and 'weight' in self.layers[i].es.attributes():
                    total_degrees.extend(list(self.layers[i].strength(weights='weight')))
                else:
                    total_degrees.extend(list(self.layers[i].degree()))
        return np.array(total_degrees)

    def get_interlayer_degrees(self):
        degrees=np.zeros(self.N)
        for i,e in enumerate(self.interlayer_edges):
            ei,ej=e[0],e[1]
            toadd=1 if self.interlayer_weights is None else self.interlayer_weights[i]
            degrees[ei]=degrees[ei]+toadd
            degrees[ej]=degrees[ej]+toadd
        return degrees

    def get_AMI_with_communities(self,labels):
        """
        Calculate adjusted mutual information of labels with underlying community of network.
        :param labels: commmunity to assess agreement with.  Len(labels) must \
        equal self.N
        :return:
        """
        if self.comm_vec is None:
            raise ValueError("Must provide communities lables for Multilayer Graph")
        return skm.adjusted_mutual_info_score(self.comm_vec,labels_pred=labels)

    def get_AMI_layer_avg_with_communities(self,labels):
        """
        Calculate AMI of each layer with corresponding community in labels.  Return \
        average AMI weighted by number of nodes in each layer.
        :param labels: commmunity to assess agreement with.  Len(labels) must \
        equal self.N
        :return:
        """
        if self.comm_vec is None:
            raise ValueError("Must provide communities lables for Multilayer Graph")

        la_amis=[]
        lay_vals=np.unique(self.layer_vec)
        for lay_val in lay_vals:
            cinds=np.where(self.layer_vec==lay_val)[0]
            la_amis.append(len(cinds)/(1.0*self.N)*skm.adjusted_mutual_info_score(labels_true=labels[cinds],labels_pred=self.comm_vec[cinds]))

        return np.sum(la_amis) #take the average weighted by number of nodes in each layer
        
    def get_accuracy_with_communities(self,labels,permute=True):
        """Calculate accuracy between supplied labels and the known communities of the networks.

        :param labels:  commmunity to assess agreement with.  Len(labels) must \
        equal self.N
        :param permute:  Should maximum accurracy across label permuations be identified?
        :return:
        """
        if self.comm_vec is None:
            raise ValueError("Must provide communities lables for Multilayer Graph")

        #TODO
        # #this needs to be re-written to be more efficient.  Can use bipartite matching here.

        if permute:
            vals=np.unique(labels)
            all_acc=[]
            ncoms=float(len(np.unique(self.comm_vec)))
            for perm in it.permutations(vals):
                cdict=dict(zip(vals,perm))
                mappedlabels=list(map(lambda x : cdict[x],labels))
                acc=skm.accuracy_score(y_pred=mappedlabels,y_true=self.comm_vec,normalize=False)
                acc=(acc-self.N/ncoms)/(self.N-self.N/ncoms)
                all_acc.append(acc)
            return np.max(all_acc) #return value with highest accuracy
        else:
            return skm.accuracy_score(y_true=self.comm_vec,y_pred=labels)

    def get_accuracy_layer_averaged_with_communities(self,labels,permute=True):
        if self.comm_vec is None:
            raise ValueError("Must provide communities lables for Multilayer Graph")

        la_amis = []
        lay_vals = np.unique(self.layer_vec)
        for lay_val in lay_vals:
            cinds = np.where(self.layer_vec == lay_val)[0]

            clabs=labels[cinds]
            ctrue=self.comm_vec[cinds]
            if permute:
                vals=np.unique(clabs)
                all_acc=[]
                ncoms=float(len(np.unique(ctrue)))
                for perm in it.permutations(vals):
                    cdict=dict(zip(vals,perm))
                    mappedlabels=list(map(lambda x : cdict[x],clabs))
                    acc=skm.accuracy_score(y_pred=mappedlabels,y_true=ctrue,normalize=False)
                    c_n=float(len(clabs)) #size of current layer
                    acc=(acc-c_n/ncoms)/(c_n-c_n/ncoms)
                    all_acc.append(acc)
                la_amis.append( (len(cinds)/(1.0*self.N))*np.max(all_acc) )
            else:
                la_amis.append( (len(cinds)/(1.0*self.N))*skm.accuracy_score(y_true=ctrue,y_pred=clabs))
        return np.sum(la_amis)

    def _to_sparse(self,intra=True):
        if intra:
            edgelist=np.array(self.intralayer_edges)
            data=self.intralayer_weights
        else:
            edgelist=np.array(self.interlayer_edges)
            data=self.interlayer_weights

        # if edgelist.shape[1]>2 : #assume data is 3rd
        #     data=edgelist[:,2]
        # else:
        #     data=np.array([1.0 for _ in range(edgelist.shape[0])])
        row_ind=edgelist[:,0]
        col_ind=edgelist[:,1]
        N=self.N

        return scispa.csr_matrix((data,(row_ind,col_ind)),shape=(N,N),dtype=float)

    def to_scipy_csr(self):
        """Create sparse matrix representations of the multilayer network.

        :return: (A_sparse,C_sparse) = interlayer adjacency , interlayer adjacency
        """

        A_sparse=self._to_sparse()
        C_sparse=self._to_sparse(intra=False)
        return (A_sparse,C_sparse)

    def create_null_adj(self):
        P=np.zeros((self.N,self.N))
        P = np.zeros((self.N, self.N))
        cind = 0
        for layer in self.layers:
            if 'weight' in layer.es.attributes():
                strength = np.array(layer.strength(weights='weight'))
                pcur = np.outer(strength, strength)
                pcur /= (2.0 * np.sum(layer.es['weight']))
            else:
                strength = np.array(layer.strength())
                pcur = np.outer(strength, strength)
                pcur /= (2.0 * np.sum(layer.degree()))

            cinds = range(cind, cind + layer.vcount())
            P[np.ix_(cinds, cinds)] = pcur
            cind += layer.vcount()

        return P

    def plot_communities(self, comvec=None, layers=None, ax=None, cmap=None):
        """
        Plot communities as an nlayers by nodes/layer heatmap.  Note this only works
        for the multiplex case where the number of nodes is fixed throughout each layer.

        :param comvec: community label for each nodes.  If none, used stored ground truth for\
        the network.
        :param layers: Subset of the layers to plot.  If None, plots all layers.
        :param ax: matplotlib.Axes to draw on
        :param cmap: color map to label communities with. Defaults to cube_helix.
        :return:
        """

        if layers is None:
            layers = np.unique(self.layer_vec)

        def get_partition_matrix(partition, layer_vec):
            # assumes partiton in same ordering for each layer
            vals = np.unique(layer_vec)
            nodeperlayer = len(layer_vec) // len(vals)
            com_matrix = np.zeros((nodeperlayer, len(vals)))
            for i, val in enumerate(vals):
                cind = np.where(layer_vec == val)[0]
                ccoms = partition[cind]
                com_matrix[:, i] = ccoms
            return com_matrix

        cinds = np.where(np.isin(self.layer_vec, layers))[0]
        if comvec is None:  # use baseline
            assert self.comm_vec is not None, "Must specify ground truth com_vec for graph"
            cpart = self.comm_vec
        else:
            cpart = comvec

        vmin = np.min(cpart)
        vmax = np.max(cpart)

        clayer_vec = self.layer_vec[cinds]
        part_mat = get_partition_matrix(cpart, clayer_vec)

        if ax is None:
            ax = plt.axes()

        if cmap is None:
            cmap = sbn.cubehelix_palette(as_cmap=True)

        ax.grid('off')
        ax.pcolormesh(part_mat, cmap=cmap, vmin=vmin, vmax=vmax)

        # numswitched = self.get_number_nodes_switched_all_layers(ind=ind, percent=True)
        # numswitched = numswitched[np.where(np.isin(self.layers_unique, layers))[0]]  # filter for layers selected
        # for i, num in enumerate(numswitched):
        #     ax.text(s="{:.2f}".format(num), x=i, y=-1, fontdict={"fontsize": 9, 'color': 'white'})

        ax.set_xticks(range(0, len(layers)))
        ax.set_xticklabels(layers)
        return ax

class MultilayerSBM(MultilayerGraph):
    """
    Subclass of MultilayerGraph to create the dynamic stochastic block model from Ghasemian et al. 2016.
    """
    def __init__(self,n,comm_prob_mat,layers=2,transition_prob=.1,block_sizes0=None,use_gcc=False):
        """

        :param n: number of nodes in each layer
        :param comm_prob_mat: probability of block connections in SBM(this is fixed across all layers)
        :param layers:  number of layers
        :param transition_prob: probability of each node changing communities from one layer to next.  \
        if transistion_prob=0 then community structure is constant across all layers.
        :param block_sizes0: Initial size of the blocks.  Default is to use even block sizes starting out.\
        block sizes at subsequent layers are determined by number of nodes that randomly transition between \
        communities
        :param use_gcc: use only giant connected component of starting SBM (option for single layer only)
        """
        self.layer_sbms = []  #this is list of GenerateGraph.RandomSBMGraph objects.
        self.nlayers=layers
        self.transition_prob=transition_prob
        self.comm_prob_mat=comm_prob_mat


        if block_sizes0 is None:
            block_sizes0 = [int(n / (1.0 * comm_prob_mat.shape[0])) for _ in range(comm_prob_mat.shape[0] - 1)]
            block_sizes0 += [n - np.sum(block_sizes0)]  # make sure it sums to one

        assert not (use_gcc and layers > 1), "use_gcc only applies to single layer network"
        #create first layer
        initalSBM = RandomSBMGraph(n=n, comm_prob_mat=comm_prob_mat, block_sizes=block_sizes0, use_gcc=use_gcc)

        self.n = initalSBM.n  # number of nodes in each layer
        self.nlayers = layers # number of layers
        self.N = self.n * self.nlayers #Total number of nodes

        self._blocks=range(self.comm_prob_mat.shape[0]) #
        #initialize the first one

        initalSBM.graph.vs['id']=np.arange(n) #set id's in order
        self.layer_sbms.append(initalSBM)

        for _ in range(layers-1):
            #create the next sbm from the previous one and add it to the list.
            self.layer_sbms.append(self._get_next_sbm(self.layer_sbms[-1]))
        self.interedges=self.get_interlayer_edgelist()
        self.intraedges=self.get_intralayer_edgelist()
        self.layer_vec=self.get_node_layer_vec()

        #some redudancy between constructors here that could be cleaned up
        super(MultilayerSBM,self).__init__(intralayer_edges=self.intraedges, interlayer_edges=self.interedges,
                       layer_vec=self.layer_vec,comm_vec=self.get_all_layers_block())

        #switch this to point to the layer_sbms igraphs
        self.layers=[ lsbm.graph for i,lsbm in enumerate(self.layer_sbms)]
        # self.layers=self.layer_sbms #we only need to store these once.

    def _get_next_sbm(self, sbm):
        """
        Generate new block values for each nodes. And then create a new SBM for the next layer
        :param sbm: current sbm
        :return: GenerateGraph.RandomSBMGraph
        """

        num_transition=np.random.binomial(self.n,self.transition_prob)
        nodes2switch=np.random.choice(np.arange(self.n),replace=False,size=num_transition)
        #generate all of the newblocks

        #choose uniformly from blocks.
        newblocks=np.random.choice(self._blocks,size=len(nodes2switch))


        #increment and decrement the block sizes accordingly
        nxtblocksize=sbm.block_sizes.copy() #make deepcopy
        inds, cnts = np.unique(newblocks, return_counts=True)
        nxtblocksize[inds]+=1 * cnts
        #count old blocks to decrement
        inds, cnts = np.unique(np.array(sbm.graph.vs['block'])[nodes2switch], return_counts=True)
        nxtblocksize[inds] += -1 * cnts #decrement by number

        next_blocks=np.array(sbm.graph.vs['block'])
        newids=np.array(sbm.graph.vs['id'])
        nxtsbm=RandomSBMGraph(n=self.n,comm_prob_mat=self.comm_prob_mat,block_sizes=nxtblocksize)
        next_blocks[nodes2switch]=newblocks #switch the blocks
        # this permutes the new ids accordingly to keep blocks in order
        perm_ids=[ x[1] for x in  sorted(zip(next_blocks,newids),key=lambda x :x[0])]

        nxtsbm.graph.vs['id']=perm_ids

        #permute the nodes to line up with ids
        nxtsbm.graph=nxtsbm.graph.permute_vertices(perm_ids)
        return nxtsbm

    def get_intralayer_adj(self):
        """
        Single adjacency matrix representing all layers.
        :return: np.array of intralayer adjacency representation
        """
        intra_adj=np.zeros((self.n*self.nlayers,self.n*self.nlayers))
        for i,layer in enumerate(self.layer_sbms):
            offset=self.n*i
            inds=np.ix_(range(offset,offset+self.n),range(offset,offset+self.n)) #index diagonal block
            intra_adj[inds]=layer.get_adjacency()
        return intra_adj

    def get_interlayer_adj(self):
        """
        Singer interlayer adjencency matrix created by connecting each node to it's equivalent\
        node in the next layer.  This is a temporal multiplex topology.
        :return: np.array of interlayer adjacency representation
        """
        #TODO this can be changed for different couplings. Right now I do only adjacent layers
        inter_adj=np.zeros((self.n*self.nlayers,self.n*self.nlayers))
        Iden_N=np.identity(self.n)
        for i in range(self.nlayers):
            if i < self.nlayers-1:
                inds=np.ix_(range(i*self.n,(i+1)*self.n),range((i+1)*self.n,(i+2)*self.n))
                inds_t=np.ix_(range((i+1)*self.n,(i+2)*self.n),range(i*self.n,(i+1)*self.n))
                inter_adj[inds]=np.array(Iden_N)
                inter_adj[inds_t]=np.array(Iden_N)

        return inter_adj

    def get_interlayer_edgelist(self):
        """
        Single list of edges giving the multilayer connections.  For this model \
        nodes are multiplex an connected to their neighboring slice identities.

        :return: np.array
        """

        interedges=np.zeros((self.n*(self.nlayers-1),2),dtype='int')
        offset=0
        for i in range(self.nlayers-1):
            cnet=self.layer_sbms[i]
            cnetnxt=self.layer_sbms[i+1]
            cedge=np.array(list(zip(range(offset,offset+cnet.n),
                               range(offset+cnet.n,offset+2*cnet.n))))
            interedges[offset:offset+cnet.n,:]=cedge

            offset+=cnet.n
        return interedges

    def get_node_layer_vec(self):
        """

        :return: np.array denoting which layer each node is in
        """
        layers=[]
        for i,net in enumerate(self.layer_sbms):
            layers.extend([i for _ in range(net.n)])
        return np.array(layers)

    def get_intralayer_edgelist(self):
        """
        Single list of edges treating the group of single layer SBM's  as a \
        surpra-adjacency format

        :return: np.array
        """
        nedges=np.sum([net.m for net in self.layer_sbms])
        intraedges=np.zeros((nedges,2),dtype='int')

        offset=0
        m_offset=0 #for indexing
        for i in np.arange(self.nlayers):
            c_layernet=self.layer_sbms[i]
            if c_layernet.m==0:
                continue #no edges in this layer
            c_elist=c_layernet.get_edgelist()
            intraedges[m_offset:m_offset+c_layernet.m,:]=np.array(c_elist)+offset
            offset+=c_layernet.n
            m_offset+=c_layernet.m
        return intraedges

    def get_all_layers_block(self):
        """
        returns a single vector with block id for each node across all of the layers
        :return:
        """
        merged_blocks=[]
        for layer in self.layer_sbms:
            merged_blocks+=layer.graph.vs['block']
        return np.array(merged_blocks)


def generate_planted_partitions_sbm(n,epsilon,c,ncoms):
    """

    :param n: The number of nodes
    :param c: total average degree for the network
    :param ep: detectability parameter, :math:`\epsilon=p_{out}/p_{in}`, where \
    p is the internal and external connection probabilities
    :param ncoms: number of communities within the network
    :return:
    """

    #based on planted partition model, we can calculated the values of
    #pin and pout using the number of nodes, average degree, and the ratio
    # of the internal and external edges

    noq=n/float(ncoms)
    pin=c/(((noq-1.0)+noq*(ncoms-1)*epsilon))
    pout = epsilon * pin

    remain=n%ncoms
    if remain>0:
        nodesperblock=[int(n/ncoms)]*ncoms
        nodesperblock=[ x+1 if k<remain else x for k,x in enumerate(nodesperblock) ]
    else:
        nodesperblock=[n/ncoms]*ncoms

    assert np.sum(nodesperblock)==n

    # prob_mat=np.identity(ncoms) * pin+(np.ones((ncoms,ncoms))-np.identity(ncoms))*pout

    prob_mat=np.identity(ncoms)*(pin-pout)+np.ones((ncoms,ncoms))*pout


    sbm = MultilayerSBM(n, comm_prob_mat=prob_mat, layers=1, block_sizes0=nodesperblock,
                                 transition_prob=0, use_gcc=True)

    return sbm


def generate_planted_partitions_dynamic_sbm(n, epsilon, c, ncoms,nlayers,eta):
    """

    :param n: The number of nodes
    :param c: total average degree for the network
    :param ep: detectability parameter, :math:`\epsilon=p_{out}/p_{in}`, where \
    p is the internal and external connection probabilities
    :param ncoms: number of communities within the network
    :param nlayers: number of layers of the dynamic stochastic block model
    :param eta: probability of each node switching community label in each layer
    :return:
    """

    # based on planted partition model, we can calculated the values of
    # pin and pout using the number of nodes, average degree, and the ratio
    # of the internal and external edges

    noq = n / float(ncoms)
    pin = c / (((noq - 1.0) + noq * (ncoms - 1) * epsilon))
    pout = epsilon * pin

    remain = n % ncoms
    if remain > 0:
        nodesperblock = [int(n / ncoms)] * ncoms
        nodesperblock = [x + 1 if k < remain else x for k, x in enumerate(nodesperblock)]
    else:
        nodesperblock = [n / ncoms] * ncoms

    assert np.sum(nodesperblock) == n

    # prob_mat=np.identity(ncoms) * pin+(np.ones((ncoms,ncoms))-np.identity(ncoms))*pout

    prob_mat=np.identity(ncoms)*(pin-pout)+np.ones((ncoms,ncoms))*pout

    dsbm = MultilayerSBM(n, comm_prob_mat=prob_mat, layers=nlayers, block_sizes0=nodesperblock,
                        transition_prob=eta, use_gcc=False)

    return dsbm