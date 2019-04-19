from __future__ import absolute_import
import numpy as np
import igraph as ig
from future.utils import iteritems,iterkeys
from collections import Hashable
from .GenerateGraphs import MultilayerGraph
import sklearn.metrics as skm
from .bp import BP_Modularity,PairVector,IntArray,IntMatrix,DoubleArray
import itertools
import pandas as pd
import scipy.optimize as sciopt
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sbn
from time import time
import warnings
import os,pickle,gzip
#import #logging
#logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=#logging.DEBUG)
##logging.basicConfig(format=':%(asctime)s:%(levelname)s:%(message)s', level=#logging.INFO)

class ModularityBP():
    """
    This is python interface class for the mulitlayer modularity BP
    """

    def __init__(self, mlgraph=None, interlayer_edgelist=None,
                 intralayer_edgelist=None, layer_vec=None,
                 accuracy_off=True, use_effective=False, comm_vec=None,
                 align_communities_across_layers=True,min_com_size=5):

        assert not (mlgraph is None) or not ( intralayer_edgelist is None and layer_vec is None)


        if mlgraph is not None:
            # this is just a single layer igraph. We create a mlgraph with empty interlayer edges
            if hasattr(mlgraph, 'get_edgelist'):
                self.graph = MultilayerGraph (intralayer_edges=np.array(mlgraph.get_edgelist()),
                                              interlayer_edges=np.zeros((0,2),dtype='int'),
                                              layer_vec=[0 for _ in range(mlgraph.vcount())])

            else:
                self.graph=mlgraph

        else:
            if interlayer_edgelist is None:
                interlayer_edgelist=np.zeros((0,2),dtype='int')
            self.graph = MultilayerGraph(intralayer_edges=intralayer_edgelist,
                                         interlayer_edges=interlayer_edgelist,layer_vec=layer_vec)

        if not comm_vec is None:
            self.graph.comm_vec = comm_vec
        self.n=self.graph.N
        self.nlayers=self.graph.nlayers
        self.totaledgeweight=self.graph.totaledgeweight
        self.intralayer_edges=self.graph.intralayer_edges
        self.interlayer_edges=self.graph.interlayer_edges
        self._cpp_intra_weights=self._get_cpp_intra_weights()
        self.layer_vec=np.array(self.graph.layer_vec)
        self._layer_vec_ia=IntArray([int(i) for i in self.layer_vec])#must be integers!
        self.layers_unique=sorted(np.unique(self.layer_vec))
        self._accuracy_off=accuracy_off #calculating permuated accuracy can be expensive for large q
        self._align_communities_across_layers=align_communities_across_layers
        self.marginals={}
        self.partitions={} # max of marginals
        self.niters={}
        self.marginal_index_to_close_marginals={} #maps each marginal index to common marginals (ie marginals that have the same information across nodes
        self.marginal_to_comm_number={} # the marginal index to the effective community number
        self.group_distances={}
        self.use_effective=use_effective
        self.nruns=0 #how many times has the BP algorithm been run.  Also serves as index for outputs
        self._permutation_vectors={} # This is used to keep track of which vertices are switched after as sweep
        #make single index
        # rm_index=pd.MultiIndex(labels=[[],[],[],[]],levels=[[],[],[],[]],names=['q','beta','resgamma','omega'])
        self.retrieval_modularities=pd.DataFrame(columns=['q','beta','resgamma','omega',
                                                         'retrieval_modularity','niters'],dtype=float)


        self._intraedgelistpv= self._get_edgelistpv()
        self._interedgelistpv= self._get_edgelistpv(inter=True)
        self.min_community_size = min_com_size  #for calculating true number of communities min number of node assigned to count.
        self._bpmod=None

    def run_modbp(self,beta,q,niter=100,resgamma=1.0,omega=1.0,reset=False):
        """

        :param beta:
        :param q:
        :param niter:
        :param resgamma:
        :param omega:
        :param reset:
        :return:
        """
        if beta==0:
            warnings.warn("beta cannot be zero.  Using 10-16")
            beta=np.power(10.0,-16)

        assert(q>0),"q must be > 0"
        q_orig=q #before collapsing
        self.retrieval_modularities.loc[self.nruns, 'q'] = q_orig
        self.retrieval_modularities.loc[self.nruns, 'beta'] = beta
        self.retrieval_modularities.loc[self.nruns, 'omega'] = omega
        self.retrieval_modularities.loc[self.nruns, 'resgamma'] = resgamma

        t=time()
        #logging.debug("Creating c++ modbp object")
        if self._bpmod is None:
            #print("Creating c++ modbp object")
            #print(list(self._cpp_intra_weights))
            self._bpmod=BP_Modularity(layer_membership=self._layer_vec_ia,
                                        intra_edgelist=self._intraedgelistpv,intra_edgeweight=self._cpp_intra_weights,
                                      inter_edgelist=self._interedgelistpv,
                                      _n=self.n, _nt= self.nlayers , q=q, beta=beta,
                                      resgamma=resgamma,omega=omega,transform=False,verbose=True)

        else:
            if self._bpmod.getBeta() != beta or reset:
                self._bpmod.setBeta(beta)
            if self._bpmod.getq() != q:
                self._bpmod.setq(q)
            if self._bpmod.getResgamma() != resgamma:
                self._bpmod.setResgamma(resgamma)
            if self._bpmod.getOmega() != omega:
                self._bpmod.setOmega(omega)
                


        if self._align_communities_across_layers:
            iters_per_run=niter #somewhat arbitrary divisor
        else:
            iters_per_run=niter # run as many times as possible on first run.
        #logging.debug('time: {:.4f}'.format(time()-t))
        t=time()
        #logging.debug('Running modbp at beta={:.3f}'.format(beta))
        converged=False
        iters=self._bpmod.run(iters_per_run)
        cmargs=np.array(self._bpmod.return_marginals())
        #logging.debug('time: {:.4f}, {:d} iterations '.format(time() - t, iters))
        t=time()

        if iters<iters_per_run:
            converged=True
        self.marginals[self.nruns]=cmargs
        #Calculate effective group size and get partitions
        #logging.debug('Combining marginals')
        self._get_community_distances(self.nruns) #sets values in method
        cpartition=self._get_partition(self.nruns,self.use_effective)
        self.partitions[self.nruns]=cpartition

        if self.use_effective:
            q_new = self._merge_communities_bp(self.nruns)
            q = q_new
        #logging.debug('time: {:.4f}'.format(time()-t))
        t=time()

        # if self._align_communities_across_layers and iters<niter:
        #     #logging.debug('aligning communities across layers')
        #     # print ("Bethe : {:.3f}, Modularity: {:.3f}".format(self._bpmod.compute_bethe_free_energy(),
        #     #                                                    self._get_retrieval_modularity(self.nruns)))
        #     nsweeps=self._perform_permuation_sweep(self.nruns) # modifies partition directly



        if self._align_communities_across_layers and iters<niter:
            #logging.debug('aligning communities across layers')
            # print ("Bethe : {:.3f}, Modularity: {:.3f}".format(self._bpmod.compute_bethe_free_energy(),
            #                                                    self._get_retrieval_modularity(self.nruns)))
            nsweeps=self._perform_permuation_sweep(self.nruns) # modifies partition directly

            #logging.debug('time: {:.4f} : nsweeps: {:d}'.format(time() - t,nsweeps))
            t = time()
            cnt=0
            while not (nsweeps==0 and converged==True) and not iters>niter:
                #plots for debugging
                #logging.debug("rerunning modbp with realigned:")
                self._switch_beliefs_bp(self.nruns)
                #can't go more than the alloted number of runs
                citers = self._bpmod.run(iters_per_run)
                iters+=citers
                if citers<iters_per_run: #it converged
                    converged=True
                #logging.debug('time: {:.4f}, {:d} iterations more. total iters: {:d}'.format(time() - t,citers,iters))
                t = time()
                cmargs = np.array(self._bpmod.return_marginals())
                self.marginals[self.nruns] = cmargs
                # Calculate effective group size and get partitions
                self._get_community_distances(self.nruns)  # sets values in method
                cpartition = self._get_partition(self.nruns, self.use_effective)
                self.partitions[self.nruns] = cpartition
                if self.use_effective:
                    q_new = self._merge_communities_bp(self.nruns)
                    q = q_new
                cnt+=1

                nsweeps = self._perform_permuation_sweep(self.nruns)  # modifies partition directly

            #final combined marginals

                # #logging.debug('time: {:.4f}'.format(time() - t))
                #logging.debug('nsweeps: {:d}'.format(nsweeps))

        # Perform the merger on the BP side before getting final marginals
        if iters>=niter:
            #logging.debug("Modularity BP did not converge after {:d} iterations.".format(iters))
            pass



        self.retrieval_modularities.loc[self.nruns, 'niters'] = iters
        self.retrieval_modularities.loc[self.nruns, 'converged'] = converged
        retmod=self._get_retrieval_modularity(self.nruns)
        #logging.debug('calculating bethe_free energy')
        bethe_energy=self._get_bethe_free_energy()
        self.retrieval_modularities.loc[self.nruns,'retrieval_modularity']=retmod
        self.retrieval_modularities.loc[self.nruns,'bethe_free_energy']=bethe_energy
        self.retrieval_modularities.loc[self.nruns,'avg_entropy']=_get_avg_entropy(cmargs)
        _,com_cnts=np.unique(self.partitions[self.nruns],return_counts=True)

        # self.retrieval_modularities.loc[self.nruns,'num_coms']=self._get_true_number_of_communities(self.nruns)
        self.retrieval_modularities.loc[self.nruns, 'num_coms']=np.sum(com_cnts>=self.min_community_size)
       # self.retrieval_modularities.loc[self.nruns,'qstar']=self._get_true_number_of_communities(self.nruns)
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

    # def _get_edgelist(self):
    #     edgelist=self.graph.get_edgelist()
    #     if not self.graph.intralayer_weights is None:
    #         elist_weights=sorted(zip(self.graph.),key=lambda (x):x[0],x[1])
    #     edgelist.sort()
    #     return edgelist


    def _get_bethe_free_energy(self):
        if self._bpmod is None:
            raise AssertionError( "cannot calculate the bethe free energy without running first.  Please call run_mobp.")
        return self._bpmod.compute_bethe_free_energy()

    def _get_cpp_intra_weights(self):
        if self.graph.unweighted:
            return DoubleArray([])
        else:
            assert len(self.graph.intralayer_weights)==len(self.graph.intralayer_edges),"length of weights must match number of edges"
            return DoubleArray(self.graph.intralayer_weights)

    def _get_edgelistpv(self,inter=False):
        ''' Return PairVector swig wrapper version of edgelist'''
        if inter:
            try: #fonud that type numpy.int
                _edgelistpv = PairVector(self.interlayer_edges) #cpp wrapper for list
            except NotImplementedError:
                self.interlayer_edges=[ (int(e[0]),int(e[1])) for e in self.interlayer_edges]
                _edgelistpv = PairVector(self.interlayer_edges)
        else:
            try:
                _edgelistpv = PairVector(self.intralayer_edges)
            except NotImplementedError:
                self.intralayer_edges = [(int(e[0]), int(e[1])) for e in self.intralayer_edges]
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
            try:
                return np.random.choice(np.flatnonzero(np.abs(x-x.max())<np.power(10.0,-6)))
            except:
                print(x)
                raise ValueError

        parts=np.apply_along_axis(func1d=argmax_breakties,arr=marginal,axis=1)


        if use_effective: #map the marginals to very close ones.
            groupmap=self.marginal_index_to_close_marginals[ind]
            # We use the effective communities to map
            parts=np.array( map(lambda x: self.marginal_to_comm_number[ind][x], parts))
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
                                      intra_edgeweight=self._cpp_intra_weights,
                                      inter_edgelist=self._interedgelistpv,
                                      _n=self.n, _nt= self.nlayers , q=q, beta=1.0, #beta doesn't matter
                                       omega=omega,transform=False)

        return self._bpmod.compute_bstar(omega,int(q)) #q must be an int

    def _get_retrieval_modularity(self,nrun=None):
        """

        :param nrun:
        :return:
        """
        if nrun is None:
            nrun=self.nruns #get last one
        resgamma,omega=self.retrieval_modularities.loc[nrun,['resgamma','omega']]
        cpartition = self.partitions[nrun] #must have already been run
        return calc_modularity(self.graph,partition=cpartition,resgamma=resgamma,omega=omega)
        

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

        #get direcly from the mariginals
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
                for val in comb: #have to update everyone's groups
                    groups[val]=comb


        self.marginal_index_to_close_marginals[ind]=groups

        commsets = list(set([frozenset(s) for s in groups.values()]))
        revmap={}
        for i,comset in enumerate(commsets):
            for val in comset:
                #we use the minimum community mapped so that must number of communities
                #will retain mapping.
                revmap[val]=np.min(list(comset))

        #remap to those in the range from 0 to len(commsets)
        available=set(np.arange(len(commsets))).difference(set(revmap.values()))
        valsremap={}
        #use this dict to ensure that all vals are remapped consistently

        for k,val in list(set(revmap.items())):
            if val >= len(commsets):
                if val not in valsremap:
                    valsremap[val]=available.pop()
                revmap[k]=valsremap[val]

        if np.max(list(revmap.values()))>=len(set(revmap.values())):
            raise AssertionError

        self.marginal_to_comm_number[ind] = revmap



    def _groupmap_to_permutation_vector(self,ind):
        """
        Create vector that denotes communities that should be collapsed . \
        Each element of the array is the new community label for that index . \
        I.e [ 0, 0 , 1 , 2] Denotes a collpase from 4 communities to 3 where \
        the 0th and 1st old communities are merged into a single set of marginals.
        :param ind:
        :return:
        """
        revgroupmap=self.marginal_to_comm_number[ind]

        outarray=np.arange(np.max(revgroupmap.keys())+1)

        for k,val in revgroupmap.items():
            outarray[k]=val

        return outarray
        # return IntArray(outarray)



    def _get_true_number_of_communities(self,ind):
        """

        :param ind:
        :return:
        """

        if ind not in self.marginal_index_to_close_marginals.keys():
            self._get_community_distances(ind)
        groupmap=self.marginal_index_to_close_marginals[ind]

        #create set of sets and take len.  Frozenset is immutable
        #
        if self.min_community_size==0:
            return len(set([frozenset(s) for s in groupmap.values()]))
        else:
            return len(set([ frozenset(s) for s in groupmap.values() if len(s) >= self.min_community_size ]))

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

    def _initialize_final_permutation_dict_all_layers(self,ind):
        """
        this object keeps track of which community every community is mapped to in each layer \
        in the permutation sweep to try and align the layers.

        :param ind:
        :return: list with one dictionary for each layer
        """
        layers=self.layers_unique
        final_permutation_dict = [ ]
        for i,layer in enumerate(layers):
            lay_inds=np.where(self.layer_vec==layer)[0]
            #all the possible community labels
            partvals=np.unique(self.marginal_to_comm_number[ind].values())
            #map to itself
            final_permutation_dict.append(dict(zip(partvals,partvals)))
        return final_permutation_dict


    def _perform_permuation_sweep(self,ind):
        """
        Calculate largest difference between adjacent layers\
        then perform flip for everylayer afterwards
        Repeat until no more flips are performed

        :param ind: partition to perform permutation on
        :return: number of sweeps performed. To keep track of whether \
        any layers were actually shuffled.
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



        #for each sweep perform we keep track of which communities are switched within each
        #layer.  We start out with each community mapped to itself.
        self._permutation_vectors[ind]=self._initialize_final_permutation_dict_all_layers(ind=ind)


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
            niters+=1
        return niters

        # plt.close()
        # f, a = plt.subplots(1, 2, figsize=(6, 3))
        # a = plt.subplot(1, 2, 1)
        # self.plot_communities(ax=a)
        # a = plt.subplot(1, 2, 2)
        # self.plot_communities(ind, ax=a)
        # plt.show()
        # dir="/Users/whweir/Documents/UNC_SOM_docs/Mucha_Lab/Mucha_Python/ModBP_gh/ModularityBP_Cpp/test"
        # with gzip.open(os.path.join(dir,'test_case_partswitch.txt'),'w') as fh:
        #     pickle.dump((self.layer_vec,max_layer_switched,oldpart),fh)
        # print(max_layer_switched,permdict)
        # print(niters)

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

    def _get_previous_layer_inds_dict(self,layer):
        """
        returns 2 dictionaries, mapping the indices of the nodes in the previous layer to that \
        in the current layer. This is used to define the adjacent layer of nodes.

        creates interlayer_edge_dict={ node1 : [nodes with interlayer connections], node2 : []}

        :return: cur2prev_inds = dict mapping from current layer id to previous layer ids based on interalyer edges \
        prev2cur = dict mapping from previous layer to current layer.  Both these are based on the interlayer \
        edges and neither are strictly one-to-one


        """
        if layer==0: #nothing in the previous layer
            return {},{}

        if "interlayer_edge_dict" not in self.__dict__:
            self.interlayer_edge_dict={}
            for e in self.interlayer_edges:
                ei,ej=e[0],e[1]
                self.interlayer_edge_dict[ei]=self.interlayer_edge_dict.get(ei,[])+[ej]
                self.interlayer_edge_dict[ej]=self.interlayer_edge_dict.get(ej,[])+[ei]

        cur_inds=np.where(self.layer_vec==layer)[0]
        prev_inds=np.where(self.layer_vec==(layer-1))[0]
        prev2cur={}
        cur2prev={}
        for ind in prev_inds:
            if ind in self.interlayer_edge_dict:
                prev2cur[ind]=list(set(self.interlayer_edge_dict[ind]).intersection(cur_inds))

        #construct mapping other direction
        for prevind in prev2cur.keys():
            for cind in prev2cur.get(prevind,[]):
                cur2prev[cind]=list(set(cur2prev.get(cind,[])+[prevind]))

        return cur2prev,prev2cur



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
        cur2prev_inds, prev2cur_inds = self._get_previous_layer_inds_dict(layer)
        prev_inds=prev2cur_inds.keys()

        curpart = self.partitions[ind][cind]
        prevpart = self.partitions[ind][prevind]
        curcoms = np.unique(curpart)
        prevcoms = np.unique(prevpart)
        distmat = np.zeros((len(prevcoms), len(curcoms)))

        # the index within the current layer partition
        prev_inds = {com: np.where(prevpart == com)[0] for com in prevcoms}
        cur_inds = {com: np.where(curpart == com)[0] for com in curcoms}

        prevcoms2_i=dict(zip(prevcoms,range(len(prevcoms))))
        curcoms2_j=dict(zip(curcoms,range(len(curcoms))))

        for prev_ind in prev2cur_inds.keys():
            pre_com=self.partitions[ind][prev_ind]
            i=prevcoms2_i[pre_com]
            for cur_ind in prev2cur_inds[prev_ind]:
                cur_com=self.partitions[ind][cur_ind]
                j=curcoms2_j[cur_com]
                #distmat[i, : ]+=(1.0/len(prev2cur_inds[prev_ind]))
                distmat[ i , : ]+=(1.0/len(prev2cur_inds[prev_ind]))
                distmat[ i , j ]-=(1.0/len(prev2cur_inds[prev_ind]))
        for cur_ind in cur2prev_inds.keys():
            cur_com = self.partitions[ind][cur_ind]
            j = curcoms2_j[cur_com]
            for prev_ind in cur2prev_inds[cur_ind]:
                prev_com = self.partitions[ind][prev_ind]
                i = prevcoms2_i[prev_com]
                # distmat[i, : ]+=(1.0/len(prev2cur_inds[prev_ind]))
                distmat[:, j] += (1.0 / len(cur2prev_inds[cur_ind]))
                distmat[i, j] -= (1.0 / len(cur2prev_inds[cur_ind]))


        #below works for multiplex, but not for more general multilayer topologies.
        # for i, prevcom in enumerate(prevcoms):
        #     for j, curcom in enumerate(curcoms):
        #         # how many of the currentl communities are not in the previous community
        #         #plus how many of the previous communities are not in the current community
        #         #have to figure out which of the nodes from previous layer are actually in this layer
        #         #(in the case where it's not multiplex)
        #         distmat[i, j] = np.sum(curpart[prev_inds[prevcom]] != curcom) + \
        #                         np.sum(prevpart[cur_inds[curcom]] != prevcom)

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
        curcoms=np.unique(self.partitions[ind]) #use all communities here.
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



        self.partitions[ind][lay_inds]=\
            map(lambda x : curpermutation[x], self.partitions[ind][lay_inds])

        #also apply map to the final permutation dictionary
        #No Communities should be merged or destroyed in this mapping
        #Each value (communities from previous permutation) should still be in the dictionary
        for k,val in self._permutation_vectors[ind][layer].items():
            if val in curpermutation: #otherwise not affected
                self._permutation_vectors[ind][layer][k]=curpermutation[val]

        assert len(set(self._permutation_vectors[ind][layer].values()))==len(self._permutation_vectors[ind][layer].values()), 'community lost in permutation'
        #sanity check.  Internal communities shouldn't change
        assert(np.abs(skm.adjusted_mutual_info_score(old_layer,self.partitions[ind][lay_inds])-1)<np.power(10.0,-6))



    def _create_all_layers_permuation_vector(self,ind):
        """
        We use the dictionary representation to create the permuation vector to pass into \
        the modularity bp\.  This is organized such that each row represents the current layer \
        And each position has the new position it is supposed to map to .  Ie [ 1 ,  ... ] \
        denotes that the zeroeth belief becomes the first belief (marginal) .
        :param ind:
        :return:
        """
        layers=self.layers_unique
        N=len(self.layers_unique)
        M=len(np.unique(self.marginal_to_comm_number[ind].values()))
        outarray=np.zeros((N,M)) #layers by #communites (after combining)

        numcoms=len(set(self.marginal_to_comm_number[ind].values()))

        for i,layer in enumerate(layers):

            currow = range(numcoms)
            #use the final mapping dictionary to map each of the communities in this layer
            currow = map ( lambda  x : self._permutation_vectors[ind][layer][x],currow)
            outarray[i,:]=currow

        return outarray

    def _merge_communities_bp(self,ind):
        """
        Merge communities that have the same marginals across all of the nodes
        :param ind:
        :return:
        """
        merge_vec=IntArray(self._groupmap_to_permutation_vector(ind).astype(int))
        self._bpmod.merge_communities(merge_vec)
        return len(set(self.marginal_to_comm_number[ind].values())) #new number of communities


    def _switch_beliefs_bp(self, ind):
        """
        This switches the belefs.  Should only be called after _perform_permutation_sweep
        :param ind: modbp run index
        :returns: permutes the beliefs or marginals
        """
        perm_vec_c=self._create_all_layers_permuation_vector(ind).astype(int)
        perm_vec_c=IntMatrix(perm_vec_c)
        self._bpmod.permute_beliefs(perm_vec_c)


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


        cinds=np.where(np.isin(self.layer_vec,layers))[0]
        if ind is None: #use baseline
            assert self.graph.comm_vec is not None, "Must specify ground truth com_vec for graph"
            cpart=self.graph.comm_vec
        else:
            cpart=self.partitions[ind][cinds]

        vmin=np.min(cpart)
        vmax=np.max(cpart)

        clayer_vec=np.array(self.layer_vec)[cinds]
        part_mat=get_partition_matrix(cpart,clayer_vec)

        if ax is None:
            ax=plt.axes()

        if cmap is None:
            cmap=sbn.cubehelix_palette(as_cmap=True)


        ax.grid('off')
        ax.pcolormesh(part_mat,cmap=cmap,vmin=vmin,vmax=vmax)

        # numswitched=self.get_number_nodes_switched_all_layers(ind=ind,percent=True)
        # # numswitched=numswitched[np.where(np.isin(self.layers_unique,layers))[0]] #filter for layers selected
        # for i,num in enumerate(numswitched):
        #     ax.text(s="{:.2f}".format(num),x=i,y=-1,fontdict={"fontsize":9,'color':'white'})

        ax.set_xticks(range(0,len(layers)))
        ax.set_xticklabels(layers)
        return ax

def _get_avg_entropy(marginal):
    """caculate normalized entropies from marginals"""
    entropies=[]
    for i in range(marginal.shape[0]):
        entropies.append(stats.entropy(marginal[i])/np.log(marginal.shape[1]))
    return np.mean(entropies)

def calc_modularity(graph,partition,resgamma,omega):
    """

    :param nrun:
    :return:
    """


    # we sort indices into alike
    com_inddict = {}
    allcoms = sorted(list(set(partition)))
    sumA = 0
    Ahat = 0
    Chat = 0

    def part_equal(x):
        if partition[int(x[0])] == partition[int(x[1])]:
            if len(x)>2:
                return x[2] #edge weights should be x[2]
            else:
                return 1.0
        else:
            return 0.0

    intra_edges=graph.intralayer_edges
    inter_edges = graph.interlayer_edges
    # zip weights into the edges
    if graph.intralayer_weights is not None:
        temp=[]
        for i,e in enumerate(intra_edges):
            temp.append((e[0],e[1],graph.intralayer_weights[i]))
        intra_edges=temp
    if graph.interlayer_weights is not None:
        temp=[]
        for i,e in enumerate(inter_edges):
            temp.append((e[0],e[1],graph.interlayer_weights[i]))
        inter_edges=temp

    # For Ahat and Chat we simply iterate over the edges and count internal ones
    if len(intra_edges) > 0:
        Ahat = np.sum(np.apply_along_axis(func1d=part_equal, arr=intra_edges, axis=1))
    else:
        Ahat = 0
    if not graph.is_directed:
        Ahat=Ahat*2.0

    if len(inter_edges) > 0:
        Chat = np.sum(np.apply_along_axis(func1d=part_equal, arr=inter_edges, axis=1))
    else:
        Chat = 0

    if not graph.is_directed:
        Chat = Chat * 2.0

    # We calculate Phat a little differently since it requires degrees of all members of each group
    # store indices for each community together in dict
    for i, val in enumerate(partition):
        try:
            com_inddict[val] = com_inddict.get(val, []) + [i]
        except TypeError:
            raise TypeError("Community labels must be hashable- isinstance(%s,Hashable): " % (str(val)), \
                            isinstance(val, Hashable))
    # convert indices stored together to np_array
    for k, val in iteritems(com_inddict):
        com_inddict[k] = np.array(val)

    Phat = 0
    degrees = graph.get_intralayer_degrees()  # get all degrees
    for i in range(graph.nlayers):
        c_layer_inds = np.where(graph.layer_vec == i)[0]
        # TODO for weighted network this should be the edge strengths
        for com in allcoms:
            cind = com_inddict[com]
            # get only the inds in this layer
            cind = cind[np.isin(cind, c_layer_inds)]
            cdeg = degrees[cind]  #
            if cind.shape[0] == 1:
                continue  # contribution is 0
            else:
                cPmat = np.outer(cdeg, cdeg.T)
                Phat += (np.sum(cPmat) / (2.0 * graph.intra_edge_counts[i]))

    mu= 2.0*graph.totaledgeweight if not graph.is_directed else graph.totaledgeweight
    return (1.0 / mu) * (Ahat - resgamma * Phat + omega * Chat)