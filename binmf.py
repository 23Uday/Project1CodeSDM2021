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

_DEF_MAXITER = 50
_DEF_INIT = 'nndsvd'
_DEF_CONV = 1e-8
_DEF_LMBDA = 0
_DEF_Ws = 1
_DEF_Wrs = 1
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None


# pdb.set_trace()
#	-------------------- This part holds functions for NMF Init --------------------
_log = logging.getLogger('NNDSVD Initialization')
def safe_vstack(Xs):
	if any(sp.issparse(X) for X in Xs):
		return sp.vstack(Xs)
	else:
		return np.vstack(Xs)


def norm(x):
	"""Dot product-based Euclidean norm implementation

	See: http://fseoane.net/blog/2011/computing-the-vector-norm/
	"""
	return math.sqrt(squared_norm(x))


def trace_dot(X, Y):
	"""Trace of np.dot(X, Y.T)."""
	return np.dot(X.ravel(), Y.ravel())


def _sparseness(x):
	"""Hoyer's measure of sparsity for a vector"""
	sqrt_n = np.sqrt(len(x))
	return (sqrt_n - np.linalg.norm(x, 1) / norm(x)) / (sqrt_n - 1)


def check_non_negative(X, whom):
	X = X.data if sp.issparse(X) else X
	if (X < 0).any():
		raise ValueError("Negative values in data passed to %s" % whom)


def _initialize_nmf(X, n_components, variant=None, eps=1e-6,
					random_state=None):
	"""NNDSVD algorithm for NMF initialization.

	Computes a good initial guess for the non-negative
	rank k matrix approximation for X: X = WH

	Parameters
	----------

	X : array, [n_samples, n_features]
		The data matrix to be decomposed.

	n_components : array, [n_components, n_features]
		The number of components desired in the approximation.

	variant : None | 'a' | 'ar'
		The variant of the NNDSVD algorithm.
		Accepts None, 'a', 'ar'
		None: leaves the zero entries as zero
		'a': Fills the zero entries with the average of X
		'ar': Fills the zero entries with standard normal random variates.
		Default: None

	eps: float
		Truncate all values less then this in output to zero.

	random_state : numpy.RandomState | int, optional
		The generator used to fill in the zeros, when using variant='ar'
		Default: numpy.random

	Returns
	-------

	(W, H) :
		Initial guesses for solving X ~= WH such that
		the number of columns in W is n_components.

	References
	----------
	C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for 
	nonnegative matrix factorization - Pattern Recognition, 2008

	http://tinyurl.com/nndsvd
	"""
	check_non_negative(X, "NMF initialization")
	if variant not in (None, 'a', 'ar'):
		raise ValueError("Invalid variant name")

	U, S, V = randomized_svd(X,n_components)
	W, H = np.zeros(U.shape), np.zeros(V.shape)

	# The leading singular triplet is non-negative
	# so it can be used as is for initialization.
	W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
	H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

	for j in range(1, n_components):
		x, y = U[:, j], V[j, :]

		# extract positive and negative parts of column vectors
		x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
		x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

		# and their norms
		x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
		x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

		m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

		# choose update
		if m_p > m_n:
			u = x_p / x_p_nrm
			v = y_p / y_p_nrm
			sigma = m_p
		else:
			u = x_n / x_n_nrm
			v = y_n / y_n_nrm
			sigma = m_n

		lbd = np.sqrt(S[j] * sigma)
		W[:, j] = lbd * u
		H[j, :] = lbd * v

	W[W < eps] = 0
	H[H < eps] = 0

	if variant == "a":
		avg = X.mean()
		W[W == 0] = avg
		H[H == 0] = avg
	elif variant == "ar":
		random_state = check_random_state(random_state)
		avg = X.mean()
		W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
		H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

	return W, H

# ----------End of NMF----------

_log = logging.getLogger('Coupled NMF')

def cnmf(D,A,rank,**kwargs):
	"""
	CNMF algorithm to compute the factorization.


	Parameters
	----------
	D : list
		List of matrices where each matrix has dimension #pix * #images.
		These matrices represent different channels of the image
		Each matrix is expected to be dense
	A : list
		List of activations for neurons of different layers
		The layers may not be consecutive in the actual neural network
		a given A[i] is of dimensions #neurons in the layer * #images
		A map from neuron number in matrix to neuron location in the layer is need in the calling function
	rank : int
		Rank of the factorization
	lmbdaF : float, optional
		Regularization parameter for F factor matrix. 0 by default
	lmbdaP : float, optional
		Regularization parameter for P_i factor matrices. 0 by default
	lmbdaO : float, optional
		Regularization parameter for O_j factor matrices. 0 by default
		for each attribute
	maxIter : int, optional
		Maximium number of iterations of the algorithm. 50 by default.
	conv : float, optional
		Stop when residual of factorization is less than conv. 1e-4 by default

	Returns
	-------
	F : ndarray
		array of shape ('N', 'rank') corresponding to the factor matrix A
	P : list
		list of matrices of shape (#pixels, 'rank') corresponding to the factorization of an image channel
	O : list
		list of matrices of shape (#neurons, 'rank') corresponding to the factorization of an activation
	f : float
		function value of the factorization
	itr : int
		number of iterations until convergence
	exectimes : ndarray
		execution times to compute the updates in each iteration


	"""


	maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
	lmbdaF = kwargs.pop('lmbdaF', _DEF_LMBDA)
	lmbdaPi = kwargs.pop('lmbdaPi', _DEF_LMBDA)
	lmbdaOj = kwargs.pop('lmbdaOj', _DEF_LMBDA)
	conv = kwargs.pop('conv', _DEF_CONV)
	compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
	# pdb.set_trace()


	print(
		'[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' %
		(rank,maxIter, conv, lmbdaF))

# ---------- initialize F ------------------------------------------------
	print('Initialization of F = NNDSVD')
	# F = _initialize_nmf(sum(D),rank)[1].transpose()
	Ft = np.random.rand(rank,D[0].shape[1])
	print('Initialized F: NNDSVD')

# ---------- initialize Pi ------------------------------------------------
	P = []
	S = []
	for i in range(len(D)):
		# print('Initialization of P[%s] = NNDSVD'%i)
		# P.append(_initialize_nmf(D[0],rank)[0])
		P.append(np.random.rand(D[0].shape[0],rank))
		S.append(np.random.rand(rank,rank))
		# print('Initialized P[%s]: NNDSVD'%i)

# ---------- initialize Oj ------------------------------------------------
	O = []
	R = []
	for i in range(len(A)):
		# print('Initialization of O[%s] = NNDSVD'%i) 
		# O.append(_initialize_nmf(A[i],rank)[0])
		O.append(np.random.rand(A[i].shape[0],rank))
		R.append(np.random.rand(rank,rank))
		# print('Initialized O[%s]: NNDSVD'%i)

	pdb.set_trace()
	fit = fitchange = fitold = f = 0

	for itr in range(maxIter):
		fitold = fit
		tic = time.time()
		Ft = _updateFt(D,A,Ft,P,O,S,R,lmbdaF)
		for index in range(len(D)):
			P[index] = _updatePEye(D[index],Ft,P[index],S[index],index,lmbdaPi)
			S[index] = _updateSEye(D[index],Ft,P[index],index,lmbdaPi)

		for index in range(len(A)):
			O[index] = _updateOJ(A[index],Ft,O[index],R[index], index,lmbdaOj)
			R[index] = _updateRj(A[index],Ft,O[index], index,lmbdaOj)


		if compute_fit:
			LOSSTOTAL, LOSSACT = _compute_RMSETOTAL(D,A,Ft,P,O,S,R)
			fit = LOSSTOTAL
		else:
			fit = itr


		toc = time.time()

		exectime = abs(toc-tic)

		fitchange = abs(fitold - fit)

		print('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
			itr, fit, fitchange, exectime
		))

		if itr > 0 and fitchange < conv:
			break




	# fitData,fitAct = _compute_fit(D,A,F,P,O)
	LOSSTOTAL, LOSSACT = _compute_RMSETOTAL(D,A,Ft,P,O,S,R)
	return Ft,P,O,S,R,LOSSTOTAL,LOSSACT




def _updateFt(D,A,Ft,P,O,S,R,lmbdaF = 0):
	# print('Updating F')
	numChannels = len(D)
	numLayers = len(A)
	ep = 1e-9
	row,col = Ft.shape
	# Computing numerator part 1

	rightMat = sum([S[i].T@P[i].T@D[i] for i in range(numChannels)]) + sum([R[j].T@O[j].T@A[j] for j in range(numLayers)])
	# leftMat = sum([S[i].T@P[i].T@P[i]@S[i] for i in range(numChannels)]) + sum([R[j].T@O[j].T@O[j]@R[j] for j in range(numLayers)]) + lmbdaF*np.eye(row)
	leftMat = sum([S[i].T@S[i] for i in range(numChannels)]) + sum([R[j].T@R[j] for j in range(numLayers)]) + lmbdaF*np.eye(row)
	Ft = np.linalg.pinv(leftMat)@rightMat
	Ft = Ft.clip(min = 0)
	
	return Ft



def _updatePEye(Di,Ft,Pi,Si,index,lmbdaPi = 0):
	# print('Updating P[%s]'%str(index))
	row,col = Pi.shape
	ep = 1e-9
	leftMat = Di@Ft.T@Si.T
	rightMat = Si@Ft@Di.T@Pi + lmbdaPi*np.eye(col)
	Pi = leftMat@np.linalg.pinv(rightMat)
	Pi = Pi.clip(min = 0)
	return Pi


def _updateSEye(Di,Ft,Pi,index,lmbdaSi = 0):
	# print('Updating P[%s]'%str(index))
	row,col = Pi.shape
	ep = 1e-9
	centerMat = Pi.T@Di@Ft.T
	rightMat = Ft@Ft.T + lmbdaSi*np.eye(col)
	Si = centerMat@np.linalg.pinv(rightMat)
	return Si


def _updateRj(Aj,Ft,Oj,index,lmbdaRj = 0):
	# print('Updating P[%s]'%str(index))
	row,col = Oj.shape
	ep = 1e-9
	# leftMat = Oj.T@Oj
	centerMat = Oj.T@Aj@Ft.T
	rightMat = Ft@Ft.T + lmbdaRj*np.eye(col)
	Rj = centerMat@np.linalg.pinv(rightMat)
	return Rj

def _updateOJ(Aj,Ft,Oj,Rj,index,lmbdaOj = 0):
	# print('Updating O[%s]'%str(index))
	row,col = Oj.shape
	ep = 1e-9
	leftMat = Aj@Ft.T@Rj.T
	rightMat = Rj@Ft@Aj.T@Oj + lmbdaOj*np.eye(col)
	Oj = leftMat@np.linalg.pinv(rightMat)
	Oj = Oj.clip(min = 0)
	return Oj

# def _compute_fit(D,A,F,P,O):
# 	fData = []
# 	for i,Di in enumerate(D):
# 		PiFt = dot(P[i],F.transpose())
# 		fData.append(1-norm(Di -PiFt)/norm(Di))


# 	fAct = []	
# 	for i,Ai in enumerate(A):
# 		OiFt = dot(O[i],F.transpose())
# 		fAct.append(1-norm(Ai -OiFt)/norm(Ai))

# 	return fData, fAct


# def _compute_fitData(D,F,P):
# 	fData = 0
# 	for i,Di in enumerate(D):
# 		PiFt = dot(P[i],F.transpose())
# 		fData+=squared_norm(Di -PiFt)/squared_norm(Di)

# 	return 1 - math.sqrt(fData)

def _compute_RMSETOTAL(D,A,Ft,P,O,S,R):
	fD = 0
	numElementsD = 0
	fA = 0
	numElementsA = 0
	baselineSum = sum([norm(mat) for mat in D])+sum([norm(mat) for mat in A])

	for i,Di in enumerate(D):
		PiSiFt = P[i]@S[i]@Ft
		fD+=squared_norm(Di -PiSiFt)/(squared_norm(Di))
		# numElementsD += Di.size

	for i,Ai in enumerate(A):
		OiRiFt = O[i]@R[i]@Ft
		fA+=squared_norm(Ai -OiRiFt)/(squared_norm(Ai))
		# numElementsA += Ai.size

	answer = (fD+fA)#/(numElementsD+numElementsA)

	return math.sqrt(answer),math.sqrt(fA)#/numElementsA)





