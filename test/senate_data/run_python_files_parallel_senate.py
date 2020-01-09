import tqdm
from contextlib import contextmanager
from multiprocessing import Pool
import itertools
import numpy as np
import os,sys
from run_senate_betascan import run_senate
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


def wrap_function_senate(args):
	'''this is to flatten further if we dont' want entirely flat argmuments'''
	output=[]
	templist=[ a if hasattr(a,'__iter__') else [a] for a in args]
	allargs=itertools.product(*templist)
	print(allargs)
	for argset in allargs:
		output.append(run_senate(*argset))
	return output


#run_louvain_multiplex_test(n,nlayers,mu,p_eta,omega,gamma,ntrials)
def run_senate_parallel():
	gamma=[.4]
	omegas=6.0
	betas=np.linspace(1.5,4,20)
	ntrials=20
	# ps = np.array([.5])
	#note the order must be correct here
	args = list(itertools.product([gamma], [omegas], betas,[ntrials]))
	output = run_parallel(wrap_function_senate, args, numprocesses=10)
	return 0


if __name__=='__main__':
    # sys.exit(run_infomap())
	sys.exit(run_senate_parallel())
