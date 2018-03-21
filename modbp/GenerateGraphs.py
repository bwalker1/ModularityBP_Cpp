import random
import numpy as np
import scipy.stats as stats
import sklearn.metrics as skm
import igraph as ig


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
    def __init__(self,n,comm_prob_mat,block_sizes=None,graph=None):

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

        self.block_sizes=np.array(block_sizes)
        self.comm_prob_mat=comm_prob_mat


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

class MultilayerSBM():

    def __init__(self,n,comm_prob_mat,layers=2,transition_prob=.1,block_sizes0=None):
        self.layer_sbms=[]
        self.n=n
        self.nlayers=layers
        self.transition_prob=.1
        self.comm_prob_mat=comm_prob_mat
        if block_sizes0 is None:
            block_sizes0 = [int(n / (1.0 * comm_prob_mat.shape[0])) for _ in range(comm_prob_mat.shape[0] - 1)]
            block_sizes0 += [n - np.sum(block_sizes0)]  # make sure it sums to one

        self._blocks=range(self.comm_prob_mat.shape[0])
        #initialize the first one
        initalSBM=RandomSBMGraph(n=self.n,comm_prob_mat=self.comm_prob_mat,block_sizes=block_sizes0)
        initalSBM.graph.vs['id']=np.arange(n) #set id's in order
        self.layer_sbms.append(initalSBM)
        for _ in range(layers-1):
            #create the next sbm from the previous one and add it to the list.
            self.layer_sbms.append(self.get_next_sbm(self.layer_sbms[-1]))
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



