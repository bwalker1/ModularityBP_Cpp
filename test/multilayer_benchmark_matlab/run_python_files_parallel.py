import tqdm
from contextlib import contextmanager
from multiprocessing import Pool
import itertools
import numpy as np
import os,sys
from run_multilayer_matlab_test_infomap import run_infomap_on_multiplex

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




def run_parallel(func,parallel_args,numprocesses=2):

	def func2pass(x):
		wrap_function(func, x)

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


def wrap_function(args):
	'''this is to flatten further if we dont' want entirely flat argmuments'''
	output=[]
	templist=[ a if hasattr(a,'__iter__') else [a] for a in args]
	allargs=itertools.product(*templist)
	print(allargs)
	for argset in allargs:
		output.append(run_infomap_on_multiplex(*argset))
	return output

# run_infomap_on_multiplex(n, nlayers, mu, p_eta, r, ntrials)

def main():
	n=100
	nlayers=10
	rs=np.append([-1],np.linspace(0,1,10))
	mus=np.linspace(0,1,10)
	ps=np.array([.5,.85,.95,.99,1])
	ntrials=2

	#we only parallelize over ps and rs
	args=list(itertools.product([n],[nlayers],[mus],ps,rs,[ntrials]))
	output=run_parallel(wrap_function,args,numprocesses=1)
	print('output')
	print(output)

if __name__=='__main__':
	sys.exit(main())