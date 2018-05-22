import numpy as np
import igraph as ig
from future.utils import iteritems,iterkeys
from collections import Hashable
from GenerateGraphs import MultilayerGraph
import sklearn.metrics as skm
from .bp import BP_Modularity,PairVector,IntArray
import itertools
import pandas as pd
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import seaborn as sbn
from time import time
import os,pickle,gzip

class ModularityBP():
    """
    This is python interface class for the mulitlayer modularity BP
    """

    def __init__(self,mlgraph=None,interlayer_edgelist=None,
                 intralayer_edgelist=None,layer_vec=None,
                 accuracy_off=False,use_effective=False,comm_vec=None,align_communities=True):

        assert not (mlgraph is None) or not ( intralayer_edgelist is None and layer_vec is None)


        if mlgraph is not None:
            # this is just a single layer igraph. We create a mlgraph with empty interlayer edges
            if hasattr(mlgraph, 'get_edgelist'):
                self.graph = MultilayerGraph (intralayer_edges=np.array(mlgraph.get_edgelist()),
                                              interlayer_edges=np.zeros((0,2),dtype='int'),
                                              layer_vec=[0 for _ in range(mlgraph.vcount())])
                if not comm_vec is None:
                    self.graph.comm_vec=comm_vec
            else:
                self.graph=mlgraph
        else:
            if interlayer_edgelist is None:
                interlayer_edgelist=np.zeros((0,2),dtype='int')
            self.graph = MultilayerGraph(intralayer_edges=intralayer_edgelist,
                                         interlayer_edges=interlayer_edgelist,layer_vec=layer_vec)
            if not comm_vec is None:
                self.graph.comm_vec = comm_vec

        self.n=self.graph.n
        self.nlayers=self.graph.nlayers
        self.totaledgeweight=self.graph.totaledgeweight
        self.intralayer_edges=self.graph.intralayer_edges
        self.interlayer_edges=self.graph.interlayer_edges
        self.layer_vec=self.graph.layer_vec
        self._layer_vec_ia=IntArray(self.layer_vec)
        self.layers_unique=sorted(np.unique(self.layer_vec))
        self._accuracy_off=accuracy_off #calculating permuated accuracy can be expensive for large q
        self._align_communities=align_communities
        self.marginals={}
        self.partitions={} # max of marginals
        self.niters={}
        self.group_maps={} #
        self.reverse_group_maps={}
        self.group_distances={}
        self.use_effective=use_effective
        self.nruns=0 #how many times has the BP algorithm been run.  Also serves as index for outputs

        #make single index
        # rm_index=pd.MultiIndex(labels=[[],[],[],[]],levels=[[],[],[],[]],names=['q','beta','resgamma','omega'])
        self.retrieval_modularities=pd.DataFrame(columns=['q','beta','resgamma','omega',
                                                         'retrieval_modularity','niters'],dtype=float)


        self._intraedgelistpv= self._get_edgelistpv()
        self._interedgelistpv= self._get_edgelistpv(inter=True)

        self._bpmod=None

    def run_modbp(self,beta,q,niter=100,resgamma=1.0,omega=1.0,reset=False,realign=True):
        """

        :param beta:
        :param q:
        :param niter:
        :param resgamma:
        :param omega:
        :param reset:
        :return:
        """
        assert(q>0),"q must be > 0"
        if self._bpmod is None:
            self._bpmod=BP_Modularity(layer_membership=self._layer_vec_ia,
                                        intra_edgelist=self._intraedgelistpv,
                                      inter_edgelist=self._interedgelistpv,
                                      _n=self.n, _nt= self.nlayers , q=q, beta=beta,
                                      resgamma=resgamma,omega=omega,transform=False,verbose=False)

        else:
            if self._bpmod.getBeta() != beta or reset:
                self._bpmod.setBeta(beta)
            if self._bpmod.getq() != q:
                self._bpmod.setq(q)
            if self._bpmod.getResgamma() != resgamma:
                self._bpmod.setResgamma(resgamma)
            if self._bpmod.getOmega() != omega:
                self._bpmod.setOmega(omega)
                
        # in case c++ class calculated b*
        if beta==0:
            beta = self._bpmod.getBeta();


        iters=self._bpmod.run(niter)


        # self._bpmod = BP_Modularity(self._edgelistpv, _n=self.n, q=q, beta=beta, transform=False)
        # iters = self._bpmod.run(niter)
        cmargs=np.array(self._bpmod.return_marginals())
        self.marginals[self.nruns]=cmargs
        # cpartition = np.argmax(cmargs, axis=1)


        #assure it is initialized
        # self.marginals[q]=self.marginals.get(q,{})
        # self.partitions[q]=self.partitions.get(q,{})
        # self.niters[q]=self.niters.get(q,{})
        # self.retrieval_modularities[q]=self.retrieval_modularities.get(q,{})

        #Calculate effective group size and get partitions
        self._get_community_distances(self.nruns) #sets values in method
        cpartition=self._get_partition(self.nruns,self.use_effective)
        self.partitions[self.nruns]=cpartition
        if self._align_communities:
            t=time()
            # print('sweeping')
            self._perform_permuation_sweep(self.nruns) # modifies partition directly
            # print('time sweeping {:.4f}'.format(time()-t))


        self.retrieval_modularities.loc[self.nruns, 'q'] = q
        self.retrieval_modularities.loc[self.nruns, 'beta'] = beta
        self.retrieval_modularities.loc[self.nruns, 'niters'] = iters
        self.retrieval_modularities.loc[self.nruns, 'omega'] = omega
        self.retrieval_modularities.loc[self.nruns, 'resgamma'] = resgamma

        retmod=self._get_retrieval_modularity(self.nruns)
        bethe_energy=self._bpmod.compute_bethe_free_energy()
        self.retrieval_modularities.loc[self.nruns,'retrieval_modularity']=retmod
        self.retrieval_modularities.loc[self.nruns,'bethe_free_energy']=bethe_energy

        _,cnts=np.unique(cpartition,return_counts=True)
        self.retrieval_modularities.loc[self.nruns,'num_coms']=np.sum(cnts>5)

        self.retrieval_modularities.loc[self.nruns,'qstar']=self._get_true_number_of_communities(self.nruns)
        self.retrieval_modularities.loc[self.nruns,'bstar']=self.get_bstar(q,omega)
        if self.graph.comm_vec is not None:
            self.retrieval_modularities.loc[self.nruns,'AMI_layer_avg']=self.graph.get_AMI_layer_avg_with_communities(cpartition)
            self.retrieval_modularities.loc[self.nruns,'AMI']=self.graph.get_AMI_with_communities(cpartition)

            if not self._accuracy_off: #for low number of communities
                self.retrieval_modularities.loc[self.nruns,'Accuracy_layer_avg']=self.graph.get_accuracy_layer_averaged_with_communities(cpartition)

                self.retrieval_modularities.loc[self.nruns, 'Accuracy'] = self.graph.get_accuracy_with_communities(cpartition)

        self.retrieval_modularities.loc[self.nruns,'is_trivial']=self._is_trivial(self.nruns)

        # self.retrieval_modularities.loc[(q,beta,resgamma,omega),'retrieval_modularity']=retmod
        # self.retrieval_modularities.loc[(q,beta,resgamma,omega),'niters']=iters
        # self.retrieval_modularities.loc[(q,beta,resgamma,omega),'AMI']=self.graph.get_AMI_with_communities(cpartition)
        # self.retrieval_modularities.sort_index(inplace=True)

        self.nruns+=1

    def _get_edgelist(self):
        edgelist=self.graph.get_edgelist()
        edgelist.sort()
        return edgelist

    def _get_edgelistpv(self,inter=False):
        ''' Return PairVector swig wrapper version of edgelist'''
        if inter:
            _edgelistpv = PairVector(self.interlayer_edges) #cpp wrapper for list
        else:
            _edgelistpv = PairVector(self.intralayer_edges)
        return _edgelistpv

    def _get_partition(self,ind,use_effective=True):
        """ We want to have argmax with randomly broken ties.

        :param ind: index of the marginal to use
        :return:
        """
        #thanks to SO 42071597

        marginal=self.marginals[ind]

        def argmax_breakties(x):
            return np.random.choice(np.flatnonzero(np.abs(x-x.max())<np.power(10.0,-6)))

        parts=np.apply_along_axis(func1d=argmax_breakties,arr=marginal,axis=1)


        if use_effective: #map the marginals to very close ones.
            groupmap=self.group_maps[ind]
            # We use the effective communities to map
            parts=np.array(map(lambda x: self.reverse_group_maps[ind][frozenset(groupmap[x])],parts ))
            return parts

        else:
            return parts

    def get_bstar(self,q,omega=0):
        #c is supposed to be the average excess degree
        # degrees=self.graph.intradegrees + self.graph.interdegrees
        # d_avg=np.mean(degrees)
        # d2=np.mean(np.power(degrees,2.0))
        # c= d2/d_avg - 1
        #c=(2.0*self.totaledgeweight/(self.n))
        # return np.log(q/(np.sqrt(c)-1)+1)

        if self._bpmod is None:
            self._bpmod=BP_Modularity(layer_membership=self._layer_vec_ia,
                                        intra_edgelist=self._intraedgelistpv,
                                      inter_edgelist=self._interedgelistpv,
                                      _n=self.n, _nt= self.nlayers , q=q, beta=1.0, #beta doesn't matter
                                       omega=omega,transform=False)
        return self._bpmod.compute_bstar(omega,q)

    def _get_retrieval_modularity(self,nrun=None):
        """

        :param nrun:
        :return:
        """
        if nrun is None:
            nrun=self.nruns #get last one

        resgamma,omega=self.retrieval_modularities.loc[nrun,['resgamma','omega']]
        cpartition = self.partitions[nrun] #must have already been run


        #we sort indices into alike
        com_inddict = {}
        allcoms = sorted(list(set(cpartition)))
        sumA = 0

        Ahat = 0
        Chat = 0

        def part_equal(x):
            if cpartition[x[0]]==cpartition[x[1]]:
                return 1
            else:
                return 0

        #For Ahat and Chat we simply iterate over the edges and count internal ones
        if self.intralayer_edges.shape[0]>0:
            Ahat=np.sum(np.apply_along_axis(func1d=part_equal, arr=self.intralayer_edges,axis=1))
        else:
            Ahat=0
        if self.interlayer_edges.shape[0]>0:
            Chat=np.sum(np.apply_along_axis(func1d=part_equal , arr=self.interlayer_edges,axis=1))
        else:
            Chat=0
        #TODO make this work for weighted edges

        # We calculate Phat a little differently since it requires degrees of all members of each group
        # store indices for each community together in dict
        for i, val in enumerate(cpartition):
            try:
                com_inddict[val] = com_inddict.get(val, []) + [i]
            except TypeError:
                raise TypeError("Community labels must be hashable- isinstance(%s,Hashable): " % (str(val)), \
                                isinstance(val, Hashable))
        # convert indices stored together to np_array
        for k, val in iteritems(com_inddict):
            com_inddict[k] = np.array(val)

        Phat=0
        degrees=self.graph.get_intralayer_degrees() #get all degrees
        for i in range(self.nlayers):
            c_layer_inds=np.where(self.graph.layer_vec==i)[0]
            #TODO for weighted network this should be the edge strengths
            for com in allcoms:
                cind = com_inddict[com]
                # get only the inds in this layer
                cind=cind[np.isin(cind,c_layer_inds)]
                cdeg=degrees[cind]#
                if cind.shape[0]==1:
                    continue #contribution is 0
                else:
                    cPmat=np.outer(cdeg,cdeg.T)
                    Phat+=(np.sum(cPmat)/(2.0*self.graph.intra_edge_counts[i]))


        return (1.0/(self.totaledgeweight))*( Ahat-resgamma*Phat+omega*Chat)

    def _get_community_distances(self,ind,thresh=np.power(10.0,-3)):
        """
        Here we calculate the average distance between the mariginals of each of the \
        communities as defined by:

        :math:`d_{l,k}=\\frac{1}{N}\\sum_{i}(\\psi_{i}^{l}-\\psi_{i}^{k})^2` \

        We also identify communities that are close enough to be considered a single community\
        i.e. their distance is below the threshhold

        :return:
        """

        try:
            cmarginal = self.marginals[ind]
        except KeyError:
            raise KeyError("Cannot find partition with index {}".format(ind))

        q=cmarginal.shape[1]

        distmat=np.zeros((q,q))

        # everyone starts out in their own group initially.
        # We merge sets together every time pairwise distance is less.
        groups=dict(zip(range(q),[{i} for i in range(q)]))


        for k,l in itertools.combinations(range(q),2):

            dist_kl=np.mean(np.power(cmarginal[:,k]-cmarginal[:,l],2.0))

            distmat[k,l]=dist_kl
            distmat[l,k]=dist_kl

            if dist_kl <=thresh:
                comb=groups[l].union(groups[k])
                groups[k]=comb
                groups[l]=comb

        self.group_maps[ind]=groups
        self.group_distances[ind]=distmat
        commsets = list(set([frozenset(s) for s in groups.values()]))
        self.reverse_group_maps[ind] = dict(zip(commsets, range(len(commsets))))  # set 2 final indice mapping



    def _get_true_number_of_communities(self,ind,min_com_size=0):
        """

        :param ind:
        :return:
        """

        if ind not in self.group_maps.keys():
            self._get_community_distances(ind)
        groupmap=self.group_maps[ind]

        #create set of sets and take len.  Frozenset is immutable
        #
        if min_com_size==0:
            return len(set([frozenset(s) for s in groupmap.values()]))
        else:
            return len(set([ frozenset(s) for s in groupmap.values() if len(s) >= min_com_size ]))

    def _is_trivial(self,ind,thresh=np.power(10.0,-3)):
        """
        We use the same metric to define marginals that represent the same partitions\
        used in _get_community_distances.

        :param ind: index of marginal to examine
        :return: true if partition is close enough to trival, false if it is sufficiently differet
        """
        cmarginal=self.marginals[ind]
        trival=np.ones(cmarginal.shape)/cmarginal.shape[1]
        if np.mean(np.power(cmarginal-trival,2.0))<thresh:
            return True
        else:
            return False

    def _get_number_switched_single_layer(self,layer,partition,percent=False):

        layers = self.layers_unique

        if layer == 0:
            return 0
        else:
            # these are the identity connection across layers
            # we check for which identities switch
            prev_layer=layers[np.where(layers == layer)[0][0] - 1]
            interedges = self.graph.interedgesbylayers[(layer, prev_layer)] #use previous layer
            num_switched = 0
            for ei, ej in interedges:
                if partition[ei] != partition[ej]:
                    num_switched += 1
            if percent:
                num_switched /= float(len(interedges))
            return num_switched

    def _perform_permuation_sweep(self,ind):
        """
        Calculate largest difference between adjacent layers\
        then perform flip for everylayer afterwards
        Repeat until no more flips are performed

        :param ind: partition to perform permutation on
        :return:
        """
        max_iters=100
        niters=0
        #go through at least once
        # number_switched = self.get_number_nodes_switched_all_layers(ind=ind, percent=True)
        # max_layer_switched = np.argmax(number_switched)
        # permdict = self._create_layer_permutation_single_layer(ind, max_layer_switched)
        # print(number_switched[max_layer_switched],max_layer_switched,permdict)
        # plt.close()
        # plt.subplots(1,2,figsize=(10,5))
        # a=plt.subplot(1,2,1)
        # a.set_title('before')
        # self.plot_communities(ind,layers=[max_layer_switched-1,
        #                                   max_layer_switched],ax=a,numbers=True)
        # for layer in range(max_layer_switched, self.layers_unique[-1] + 1):  # permute all layers behind
        #     self._permute_layer_with_dict(ind, layer=layer, permutation=permdict)
        # a = plt.subplot(1, 2, 2)
        # a.set_title('after')
        # self.plot_communities(ind, layers=[max_layer_switched - 1,
        #                                    max_layer_switched], ax=a, numbers=True)
        # plt.show()



        while niters<max_iters:
        # for clayer in self.layers_unique:
            number_switched = self.get_number_nodes_switched_all_layers(ind=ind, percent=True)
            max_layer_switched=np.argmax(number_switched)
            permdict=self._create_layer_permutation_single_layer(ind,max_layer_switched)
            # print (number_switched)
            # print(number_switched[max_layer_switched], max_layer_switched, permdict)
            # oldpart=self.partitions[ind] #before switch
            # plt.close()
            # plt.subplots(1, 2, figsize=(10, 5))
            # a = plt.subplot(1, 2, 1)
            # a.set_title('before')
            # self.plot_communities(ind, layers=[max_layer_switched - 1,
            #                                    max_layer_switched], ax=a)
            # self._permute_layer_with_dict(ind, layer=max_layer_switched, permutation=permdict)
            # a = plt.subplot(1, 2, 2)
            # a.set_title('after')
            # self.plot_communities(ind, layers=[max_layer_switched - 1,
            #                                    max_layer_switched], ax=a)
            # plt.show()
            # #
            # dir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/test"
            # with gzip.open(os.path.join(dir,'test_case_partswitch.txt'),'w') as fh:
            #     pickle.dump((self.layer_vec,max_layer_switched,oldpart),fh)
            # print(max_layer_switched,permdict)
            #
            # print()
            if all([k==v for k,v in permdict.items()]):
                break #nothing changed
            for layer in range(max_layer_switched,self.layers_unique[-1]+1): #permute all layers behind
                self._permute_layer_with_dict(ind,layer=layer,permutation=permdict)



            #after sweept





            niters+=1

        plt.close()
        f, a = plt.subplots(1, 2, figsize=(6, 3))
        a = plt.subplot(1, 2, 1)
        self.plot_communities(ax=a)
        a = plt.subplot(1, 2, 2)
        self.plot_communities(ind, ax=a)
        plt.show()
        # dir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/test"
        # with gzip.open(os.path.join(dir,'test_case_partswitch.txt'),'w') as fh:
        #     pickle.dump((self.layer_vec,max_layer_switched,oldpart),fh)
        # print(max_layer_switched,permdict)
        print(niters)

    def get_number_nodes_switched_all_layers(self, ind, percent=False):
        """
        For each layer, how many nodes switched from the previous layer ( 0 for first layer).\

        :param ind: index of the partitions to check for
        :param percent: = return percentage switched instead of number
        :return: array
        """

        cpart=self.partitions[ind]
        layers=self.layers_unique
        switched=np.zeros(len(layers))

        for i,layer in enumerate(layers):
            switched[i]=self._get_number_switched_single_layer(layer,cpart,percent=percent)

        return switched

    # def _create_com_dist_map_to_minimize(self,ind,layer):
    #     layers=self.layers_unique
    #     prev_layer = layers[np.where(layers == layer)[0][0] - 1]
    #     cind=np.where(self.layer_vec==layer)[0]
    #     prevind=np.where(self.layer_vec==prev_layer)[0]
    #
    #     curpart=self.partitions[ind][cind]
    #     prevpart=self.partitions[ind][prevind]
    #     curcoms=np.unique(curpart)
    #     prevcoms = np.unique(prevpart)
    #     distmat=np.zeros((len(prevcoms),len(curcoms)))
    #     #the index within the current layer partition
    #     prev_inds={ com:np.where(prevpart==com)[0] for com in prevcoms }
    #     for i,prevcom in enumerate(prevcoms):
    #         for j,curcom in enumerate(curcoms):
    #             #how many of the currentl communities are not in the previous community
    #             distmat[i,j]=np.sum(curpart[prev_inds[prevcom]]!=curcom)
    #     return distmat

    def _create_layer_permutation_single_layer(self,ind,layer):
        """
        Identify the permutation of community labels that minimizes the number\
        switched at the specified layer

        :param ind:
        :return:
        """

        cind = np.where(self.layer_vec == layer)[0]
        layers=self.layers_unique
        #we switch only the communiites in that layer
        layer_inds=np.where(self.layer_vec==layer)[0]
        prev_layer = layers[np.where(layers == layer)[0][0] - 1]
        prevind = np.where(self.layer_vec == prev_layer)[0]
        curpart = self.partitions[ind][cind]
        prevpart = self.partitions[ind][prevind]
        curcoms = np.unique(curpart)
        prevcoms = np.unique(prevpart)
        distmat = np.zeros((len(prevcoms), len(curcoms)))

        # the index within the current layer partition
        prev_inds = {com: np.where(prevpart == com)[0] for com in prevcoms}
        cur_inds = {com: np.where(curpart == com)[0] for com in curcoms}

        for i, prevcom in enumerate(prevcoms):
            for j, curcom in enumerate(curcoms):
                # how many of the currentl communities are not in the previous community
                #plus how many of the previous communities are not in the current community
                distmat[i, j] = np.sum(curpart[prev_inds[prevcom]] != curcom) + \
                                np.sum(prevpart[cur_inds[curcom]] != prevcom)

        #solve bipartite min cost matching with munkre algorithm
        row_ind,col_ind=sciopt.linear_sum_assignment(distmat)
        colcoms=map(lambda x : curcoms[x],col_ind)
        rwcoms=map(lambda x : prevcoms[x],row_ind)
        com_map_dict=dict(zip(colcoms,rwcoms)) #map to current layer coms to previous ones

        #Mapping needs to be one-to-one so we have to fill in communities which weren't mapped
        coms_remaining=set(curcoms).difference(com_map_dict.values())
        comsnotmapped=set(curcoms).difference(com_map_dict.keys())
        #things that are in both get mapped to themselves first
        for com in coms_remaining.intersection(comsnotmapped):
            com_map_dict[com]=com
            coms_remaining.remove(com)
            comsnotmapped.remove(com)
        for com in comsnotmapped:
            com_map_dict[com]=coms_remaining.pop()


        return com_map_dict

        #found more efficient method above
        # com_map_dict=dict(zip(coms,coms) )
        # remaining = [com for com in coms]
        # for i, com in enumerate(coms_by_size):
        #     minchange=1
        #     mincom=com
        #     for com2test in remaining:
        #         com_map_dict[com]=com2test
        #         #swap out with current changes
        #         cpart[layer_inds]=map(lambda x : com_map_dict[x],cpart[layer_inds])
        #         new_change=self._get_number_switched_single_layer(layer,partition=cpart,percent=True)
        #         if new_change<minchange:
        #             minchange=new_change
        #             mincom=com2test
        #         cpart=self.partitions[ind].copy() #copy back over from original
        #     com_map_dict[com]=mincom # use the new minimum swap
        #     remaining.remove(mincom)
        # assert(len(com_map_dict.values())==len(set(com_map_dict.values())))
        # assert(len(com_map_dict.keys())==len(coms_by_size)) #all communities must be present
        # print(com_map_dict)
        # return com_map_dict

    def _permute_layer_with_dict(self,ind,layer,permutation):
        """

        Swap a given layer by the partition dictionary.  Any community \
        not present in dictionary is mapped to itself

        :param ind: which partition to permute
        :param layer: the layer that needs to be permuated in the
        :param permutation: dictionary mapping current values to new permuted community values
        :return: none

        """


        lay_inds=np.where(self.layer_vec==layer)[0]
        old_layer=self.partitions[ind][lay_inds]
        curcoms=np.unique(old_layer)
        curpermutation={k:v for k,v in permutation.items()}

        #ensure that things don't get map to that aren't mapped so some other community
        comsremain=set(list(curcoms)+permutation.keys()).difference(permutation.values())
        coms2match=set(curcoms).difference(curpermutation.keys()) #communities that need to be matched

        for com in comsremain.intersection(coms2match) : # these map to themselve
            curpermutation[com]=com
            comsremain.remove(com)
            coms2match.remove(com)

        for com in coms2match:
            curpermutation[com]=comsremain.pop()





        # for com in curcoms:
        #     if com not in curpermutation.keys():
        #         if com in comsremain:
        #             curpermutation[com]=com
        #             comsremain.remove(com)
        #         else:
        #             curpermutation[com]=comsremain.pop()

        self.partitions[ind][lay_inds]=\
            map(lambda x : curpermutation[x], self.partitions[ind][lay_inds])

        #sanity check.  Internal communities shouldn't change
        assert(np.abs(skm.adjusted_mutual_info_score(old_layer,self.partitions[ind][lay_inds])-1)<np.power(10.0,-6))
        # stack=np.vstack([old_layer,self.partitions[ind][lay_inds]])






    def _switch_marginals(self,permutation_vector):
        """

        :param permutation_vector:
        :return:
        """
        perm_vec_c=PairVector(permutation_vector)
        self._bpmod.shuffleBeliefs(perm_vec_c)


    def plot_communities(self,ind=None,layers=None,ax=None,cmap=None):
        """

        :param ind:
        :param layers:
        :return:
        """


        if layers is None:
            layers=self.layers_unique


        def get_partition_matrix(partition, layer_vec):
            # assumes partiton in same ordering for each layer
            vals = np.unique(layer_vec)
            nodeperlayer = len(layer_vec) / len(vals)
            com_matrix = np.zeros((nodeperlayer, len(vals)))
            for i, val in enumerate(vals):
                cind = np.where(layer_vec == val)[0]
                ccoms = partition[cind]
                com_matrix[:, i] = ccoms
            return com_matrix


        cinds=np.where(np.isin(self.layer_vec,layers))
        if ind is None: #use baseline
            assert self.graph.comm_vec is not None, "Must specify ground truth com_vec for graph"
            cpart=self.graph.comm_vec
        else:
            cpart=self.partitions[ind][cinds]

        vmin=np.min(cpart)
        vmax=np.max(cpart)

        clayer_vec=self.layer_vec[cinds]
        part_mat=get_partition_matrix(cpart,clayer_vec)

        if ax is None:
            ax=plt.axes()

        if cmap is None:
            cmap=sbn.cubehelix_palette(as_cmap=True)


        ax.grid('off')
        ax.pcolormesh(part_mat,cmap=cmap,vmin=vmin,vmax=vmax)



        ax.set_xticks(range(0,len(layers)))
        ax.set_xticklabels(layers)
        return ax