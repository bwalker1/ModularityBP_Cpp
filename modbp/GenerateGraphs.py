import numpy as np
import sklearn.metrics as skm
import igraph as ig
import itertools  as it


class RandomGraph():
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

    def __init__(self,n,p):
        self.graph=ig.Graph.Erdos_Renyi(n=n,p=p,directed=False,loops=False)

# class RandomERGraph():
#
#     def __init__(self,n,p,node_inds=None):
#         """We pick the number of edges as binomial"""
#
#         num_edges=np.random.binomial(n*(n-1),p)
#         edgelist=set([])
#         if node_inds is None:
#             node_inds=np.arange(n)
#         self.m=num_edges
#         self.n=n
#         self.edgelist=np.random.choice(node_inds ,replace=True,size=(2,num_edges))
#             # edge= (i,j) if i<j else (j,i)
#             # edgelist.add(edge)
#
#         self.edgelist.sort


class RandomSBMGraph(RandomGraph):
    def __init__(self,n,comm_prob_mat,block_sizes=None,graph=None,use_gcc=False):
        """

        :param n:
        :param comm_prob_mat:
        :param block_sizes:
        :param graph:
        :param use_gcc:
        """
        if block_sizes is None:
            block_sizes = [int(n / (1.0 * comm_prob_mat.shape[0])) for _ in range(comm_prob_mat.shape[0] - 1)]
            block_sizes += [n - np.sum(block_sizes)]  # make sure it sums to one

        if graph is not None:
            self.graph=graph
        else:

            try:
                comm_prob_mat=comm_prob_mat.tolist()
            except TypeError:
                pass
            self.graph = ig.Graph.SBM(n=n, pref_matrix=comm_prob_mat, block_sizes=list(block_sizes), directed=False,loops=False)

        self.use_gcc=use_gcc
        self.block_sizes=np.array(block_sizes)
        self.comm_prob_mat=comm_prob_mat


            #nodes are assigned on bases of block
        block=[]
        for i,blk in enumerate(block_sizes):
            block+=[i for _ in range(blk)]
        self.graph.vs['block']=block

        if use_gcc==True: #in this case the block sizes need ot be recalculated
            self.graph=self.graph.components().giant()
            _,cnts=np.unique(self.graph.vs['block'],return_counts=True)
            self.block_sizes=cnts

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

    def get_pin_pout_ratio(self):
        """

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

    def get_accuracy(self, labels):
        """
        :param labels:
        :type labels:
        :return:
        :rtype:
        """
        return skm.accuracy_score(labels, self.block)

class MultilayerGraph():
    """ """
    def __init__(self,intralayer_edges,layer_vec,interlayer_edges=None,comm_vec=None):


        self.n=len(layer_vec)
        self.intralayer_edges=intralayer_edges
        if interlayer_edges is None: #Assume that it is single layer
            self.interlayer_edges=np.zeros((0,2),dtype='int')
        else:
            self.interlayer_edges=interlayer_edges

        self.layer_vec=np.array(layer_vec)
        self.layers=self._create_layer_graphs()
        self.nlayers=len(self.layers)
        self.intradegrees=self.get_intralayer_degrees()
        self.interdegrees=self.get_interlayer_degrees()
        self.intra_edge_counts=self.get_layer_edgecounts()
        self.totaledgeweight=np.sum(self.interdegrees)/2.0+np.sum(self.intradegrees)/2.0

        self.comm_vec=comm_vec #for known community labels of nodes
        if self.comm_vec is not None:
            self._label_layers(self.comm_vec)
    def _create_layer_graphs(self):
        layers=[]
        uniq=np.unique(self.layer_vec)
        for val in uniq:
            node_inds=np.where(self.layer_vec==val)[0]
            min_ind=np.min(node_inds)
            node_inds=set(node_inds) # hash for look up
            celist=[]
            #subtract this off so that number of nodes created in igraph is correct
            for ei,ej in self.intralayer_edges:
                if ei in node_inds or ej in node_inds:
                    celist.append((ei-min_ind,ej-min_ind))
            layers.append(self._create_graph_from_elist(len(node_inds),celist))
        return layers


    def _create_graph_from_elist(self,n,elist,simplify=True):
        cgraph=ig.Graph(n=n,edges=elist,directed=False)
        if simplify:
            cgraph=cgraph.simplify(multiple=True)
        return cgraph

    def _label_layers(self,comvec=None):
        """
        Here we set the true community assignmetn for each node in each layer using commvec
        :return: None
        """
        if comvec is None:
            assert self.comm_vec is not None, "Cannot set node communities if they are not provided"
            comvec=self.comm_vec

        assert len(comvec)==self.n,"length of comvec: {:d} does not equal number of nodes: {:}".format(len(comvec),self.n)
        coffset=0 #keep track of nodes already seen
        for layer in self.layers:
            layer.vs['block']=comvec[coffset:layer.vcount()]
            coffset+=layer.vcount()



    def get_layer_edgecounts(self):
        ecounts=[]
        for i in range(self.nlayers):
            ecounts.append(np.sum(self.get_intralayer_degrees(i)))
        return np.array(ecounts)

    def get_intralayer_degrees(self, i=None):
        if i is not None:
            return np.array(self.layers[i].degree())
        else:
            total_degrees=[]
            for i in range(len(self.layers)):
                total_degrees.extend(list(self.layers[i].degree()))
            return np.array(total_degrees)

    def get_interlayer_degrees(self):
        degrees=np.zeros(self.n)
        for ei,ej in self.interlayer_edges:
            degrees[ei]=degrees[ei]+1
            degrees[ej]=degrees[ej]+1
        return degrees

    def get_AMI_with_communities(self,labels):
        if self.comm_vec is None:
            raise ValueError("Must provide communities lables for Multilayer Graph")
        return skm.adjusted_mutual_info_score(self.comm_vec,labels_pred=labels)

    def get_AMI_layer_avg_with_communities(self,labels):
        if self.comm_vec is None:
            raise ValueError("Must provide communities lables for Multilayer Graph")

        la_amis=[]
        lay_vals=np.unique(self.layer_vec)
        for lay_val in lay_vals:
            cinds=np.where(self.layer_vec==lay_val)[0]
            la_amis.append(len(cinds)/(1.0*self.n)*skm.adjusted_mutual_info_score(labels_true=labels[cinds],
                                                                     labels_pred=self.comm_vec[cinds]))

        return np.sum(la_amis) #take the average weighted by number of nodes in each layer
        
    def get_accuracy_with_communities(self,labels,permute=True):
        if self.comm_vec is None:
            raise ValueError("Must provide communities lables for Multilayer Graph")

        if permute:
            vals=np.unique(labels)
            all_acc=[]
            ncoms=float(len(np.unique(self.comm_vec)))
            for perm in it.permutations(vals):
                cdict=dict(zip(vals,perm))
                mappedlabels=list(map(lambda x : cdict[x],labels))
                acc=skm.accuracy_score(y_pred=mappedlabels,y_true=self.comm_vec,normalize=False)
                acc=(acc-self.n/ncoms)/(self.n-self.n/ncoms)
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
                la_amis.append( (len(cinds)/(1.0*self.n))*np.max(all_acc) )
            else:
                la_amis.append( (len(cinds)/(1.0*self.n))*skm.accuracy_score(y_true=ctrue,y_pred=clabs))
        return np.sum(la_amis)

class MultilayerSBM():

    def __init__(self,n,comm_prob_mat,layers=2,transition_prob=.1,block_sizes0=None,use_gcc=False):

        self.layer_sbms=[]
        self.nlayers=layers
        self.transition_prob=transition_prob
        self.comm_prob_mat=comm_prob_mat


        if block_sizes0 is None:
            block_sizes0 = [int(n / (1.0 * comm_prob_mat.shape[0])) for _ in range(comm_prob_mat.shape[0] - 1)]
            block_sizes0 += [n - np.sum(block_sizes0)]  # make sure it sums to one

        assert not (use_gcc and layers > 1), "use_gcc only applies to single layer network"
        initalSBM = RandomSBMGraph(n=n, comm_prob_mat=comm_prob_mat, block_sizes=block_sizes0, use_gcc=use_gcc)

        self.n = initalSBM.n  # number of total nodeslayers
        self.nlayers = layers
        self.N = self.n * self.nlayers

        self._blocks=range(self.comm_prob_mat.shape[0]) #
        #initialize the first one

        initalSBM.graph.vs['id']=np.arange(n) #set id's in order
        self.layer_sbms.append(initalSBM)

        for _ in range(layers-1):
            #create the next sbm from the previous one and add it to the list.
            self.layer_sbms.append(self.get_next_sbm(self.layer_sbms[-1]))
        self.interedges=self.get_interlayer_edgelist()
        self.intraedges=self.get_intralayer_edgelist()
        self.layer_vec=self.get_node_layer_vec()
        # self.intra_layer_adj=self._get_intralayer_adj()
        # self.inter_layer_adj=self._get_interlayer_adj()

    def get_next_sbm(self,sbm):
        """
        Generate new block values for each nodes. And then create a new SBM for the next layer
        :param sbm:
        :return:
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
        intra_adj=np.zeros((self.n*self.nlayers,self.n*self.nlayers))
        for i,layer in enumerate(self.layer_sbms):
            offset=self.n*i
            inds=np.ix_(range(offset,offset+self.n),range(offset,offset+self.n)) #index diagonal block
            intra_adj[inds]=layer.get_adjacency()
        return intra_adj

    def get_interlayer_adj(self):
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

        :return:
        """

        interedges=np.zeros((self.n*(self.nlayers-1),2),dtype='int')
        offset=0
        for i in range(self.nlayers-1):
            cnet=self.layer_sbms[i]
            cnetnxt=self.layer_sbms[i+1]
            cedge=np.array(zip(range(offset,offset+cnet.n),
                               range(offset+cnet.n,offset+2*cnet.n)))
            interedges[offset:offset+cnet.n,:]=cedge

            offset+=cnet.n
        return interedges

    def get_node_layer_vec(self):

        layers=[]
        for i,net in enumerate(self.layer_sbms):
            layers.extend([i for _ in range(net.n)])
        return np.array(layers)

    def get_intralayer_edgelist(self):
        """
        Single list of edges treating the network as a surpaadjacency format

        :return:
        """
        nedges=np.sum([net.m for net in self.layer_sbms])
        intraedges=np.zeros((nedges,2),dtype='int')

        offset=0
        m_offset=0 #for indexing
        for i in np.arange(self.nlayers):
            c_layernet=self.layer_sbms[i]
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


