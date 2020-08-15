import logging
import time
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from numpy.random import rand
import math
import scipy.sparse as sp
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
import pdb
from scipy import linalg
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from joblib import Parallel, delayed  
import multiprocessing
import timeit
# from binmf import cnmf
from cnmf import cnmf
import os
import seaborn as sns;sns.set()



def floatMatrixToHeatMapSNS(matrix,saveAs,dpi = 1200,annot = False,linewidths = 0,fmt="f"):#Pyplot
	ax = sns.heatmap(matrix,cmap = 'hot', annot = annot,linewidths = linewidths,fmt=fmt)
	plt.savefig(saveAs,dpi = dpi)
	plt.clf()

saveAs = os.path.join(os.getcwd(),'matrix.png')

D = []
D.append(np.random.rand(10,10))
D.append(np.random.rand(10,10))
D.append(np.random.rand(10,10))
A = []
A.append(np.random.rand(10,10))
A.append(np.random.rand(10,10))
A.append(np.random.rand(10,10))
A.append(np.random.rand(10,10))
F,P,O,LOSSTOTAL,LOSSACT=cnmf(D,A,10,P_init = 'random',numGroups = 2,lmbdaF = 0.25,maxIter = 10,compute_fit = True)
floatMatrixToHeatMapSNS(F,saveAs)
