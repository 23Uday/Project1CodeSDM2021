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
# from joblib import Parallel, delayed  
import multiprocessing
import timeit
# from binmf import cnmf
from cnmf import cnmf


D = []
D.append(np.random.rand(10,10))
D.append(np.random.rand(10,10))
D.append(np.random.rand(10,10))
A = []
A.append(np.random.rand(10,10))
A.append(np.random.rand(10,10))
A.append(np.random.rand(10,10))
A.append(np.random.rand(10,10))
cnmf(D,A,8,9,maxIter = 10000000,compute_fit = True)

