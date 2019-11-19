import tqdm
from contextlib import contextmanager
from multiprocessing import Pool
import itertools
import numpy as np
import os,sys
from run_multilayer_matlab_test_infomap import run_infomap_on_multiplex
from run_multilayer_matlab_test import run_louvain_multiplex_test
@contextmanager
def terminating(obj):
	'''
	Context manager to handle appropriate shutdown of processes
	:param obj: obj to open
	:return:
	'''
	try:
		yield obj
	finally:
		obj.terminate()

## FOR RUNNING PARALLEL Experiments on the desktop



def run_parallel(func,parallel_args,numprocesses=2):


	outputlist=[]
	with terminating(Pool(processes=numprocesses)) as pool:
		tot = len(parallel_args)
		with tqdm.tqdm(total=tot) as pbar:
			# parts_list_of_list=pool.imap(_parallel_run_leiden_multimodularity,args)
			for i, res in tqdm.tqdm(enumerate(pool.imap( func, parallel_args)),
									miniters=tot):
				# if i % 100==0:
				pbar.update()
				outputlist.append(res)
	return outputlist


def wrap_function_infomap(args):
	'''this is to flatten further if we dont' want entirely flat argmuments'''
	output=[]
	templist=[ a if hasattr(a,'__iter__') else [a] for a in args]
	allargs=itertools.product(*templist)
	for argset in allargs:
		output.append(run_infomap_on_multiplex(*argset))
	return output

def wrap_function_infomap(args):
	'''this is to flatten further if we dont' want entirely flat argmuments'''
	output=[]
	templist=[ a if hasattr(a,'__iter__') else [a] for a in args]
	allargs=itertools.product(*templist)
	for argset in allargs:
		output.append(run_infomap_on_multiplex(*argset))
	return output

def wrap_function_louvain(args):
	'''this is to flatten further if we dont' want entirely flat argmuments'''
	output=[]
	templist=[ a if hasattr(a,'__iter__') else [a] for a in args]
	allargs=itertools.product(*templist)
	print(allargs)
	for argset in allargs:
		output.append(run_louvain_multiplex_test(*argset))
	return output




# run_infomap_on_multiplex(n, nlayers, mu, p_eta, r, ntrials)

def run_infomap():
	n=1000
	nlayers=15
	rs=np.append([-1],np.linspace(0,1,11))
	mus=np.linspace(0,1,11)
	ps=np.array([.5,.85,.95,.99,1])
	ntrials=20
	#we only parallelize over ps and rs
	args=list(itertools.product([n],[nlayers],[mus],ps,rs,[ntrials]))
	output=run_parallel(wrap_function_infomap,args,numprocesses=10)
	return 0

#run_louvain_multiplex_test(n,nlayers,mu,p_eta,omega,gamma,ntrials)
def run_multiplex_modularity():
	n = 1000
	nlayers = 15
	gamma=1.0
	omegas=np.append([0],np.logspace(-2.5,.5,8))
	mus = np.linspace(0, 1, 11)
	ps = np.array([.5, .85, .95, .99, 1])
	# ps = np.array([.5])
	ntrials = 100
	#note the order must be correct here
	args = list(itertools.product([n], [nlayers], [mus], ps, omegas,[gamma],[ntrials]))
	output = run_parallel(wrap_function_louvain, args, numprocesses=10)
	return 0


if __name__=='__main__':
	sys.exit(run_multiplex_modularity())