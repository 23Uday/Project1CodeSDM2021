import logging
import time
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from numpy.random import rand
import math
import scipy
import scipy.sparse as sp
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
import pdb
from scipy import linalg
import scipy.linalg.interpolative as sli
from scipy.ndimage.filters import laplace
# from joblib import Parallel, delayed  
import multiprocessing
import timeit
_DEF_NUMGROUPS = 1
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




def IDLeft(matrix,rank,random = True):
	idx,proj = sli.interp_decomp(matrix,rank,rand = random)
	Askel = sli.reconstruct_skel_matrix(matrix,rank,idx)
	return Askel

# ----------End of NMF----------

_log = logging.getLogger('Coupled NMF')

def cnmf(D,A,rank1,P_init,groupSparseF = True,**kwargs):
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

	numGroups = kwargs.pop('numGroups', _DEF_NUMGROUPS)
	maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
	lmbdaSR = kwargs.pop('lmbdaSR', _DEF_LMBDA)
	lmbdaSimplex = kwargs.pop('lmbdaSimplex', _DEF_LMBDA)
	lmbdaF = kwargs.pop('lmbdaF', _DEF_LMBDA)
	lmbdaTV = kwargs.pop('lmbdaTV', _DEF_LMBDA)
	k = kwargs.pop('k',0)
	lmbdaPi = kwargs.pop('lmbdaPi', _DEF_LMBDA)
	lmbdaOj = kwargs.pop('lmbdaOj', _DEF_LMBDA)
	conv = kwargs.pop('conv', _DEF_CONV)
	compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)


	print(
		'[Config] rank1: %d |numGroups: %d | maxIter: %d | conv: %7.1e | lmbdaF: %7.1e | lmbdaTV: %7.1e | k: %7.1e' %
		(rank1, numGroups, maxIter, conv, lmbdaF, lmbdaTV, k))

# ---------- initialize F ------------------------------------------------
	# print('Initialization of F = NNDSVD')
	# F = _initialize_nmf(sum(D),rank2)[1] # F is going to be latent component x number of examples
	F = np.random.rand(A[0].shape[1],rank1).T # get input as the number of examples
	# F = F/F.max(axis =0)
	check_non_negative(F,'Initialization of F = NNDSVD')
	# print('Initialized F: NNDSVD')

# ---------- initialize Pi ------------------------------------------------
	P = []
	# U = []
	# sigma = []
	# V = []
	for i in range(len(D)):
		if P_init == 'NNDSVD':
			print('Initialization of P[%s] = NNDSVD'%i)
			P.append(_initialize_nmf(D[i],rank1)[0])
		elif P_init == 'ID':
			P.append(IDLeft(D[i],rank1)) # ID init
			print('Initialization of P[%s] = ID'%i)
		else:
			P.append(np.random.rand(D[0].shape[0],rank1))
			print('Initialization of P[%s] = random'%i)
		check_non_negative(P[i], 'update P[%d]'%i)
		
		# U.append(np.random.rand(rank1,midRank))
		# check_non_negative(U[i],'update U[%d]'%i)

		# sigma.append(np.eye(midRank,midRank)) # diagonal
		# check_non_negative(sigma[i],'update sigma[%d]'%i)

		# V.append(np.random.rand(midRank,rank2))
		# check_non_negative(V[i],'update V[%d]'%i)
		# # print('Initialized P[%s]: NNDSVD'%i)

	# S = [U[i]@sigma[i]@V[i] for i in range(len(D))]


# ---------- initialize Oj ------------------------------------------------
	O = []
	# L = []
	# T = []
	# W = []
	k = k
	S = np.ones((rank1,rank1)) - np.eye(rank1)

	for i in range(len(A)):
		# print('Initialization of O[%s] = NNDSVD'%i)
		# O.append(_initialize_nmf(A[i],rank1)[0])
		# O.append(IDLeft(A[i],rank1,A[i].shape[0]))
		O.append(np.random.rand(A[i].shape[0],rank1))
		check_non_negative(O[i],'update O[%d]'%i)
		
		# L.append(np.random.rand(rank1,midRank))
		# check_non_negative(L[i],'update L[%d]'%i)

		# T.append(np.eye(midRank,midRank))
		# check_non_negative(T[i],'update T[%d]'%i)

		# W.append(np.random.rand(midRank,rank2))
		# check_non_negative(W[i],'update R[%d]'%i)
		# # print('Initialized O[%s]: NNDSVD'%i)

	# R = [L[i]@T[i]@W[i] for i in range(len(A))]

	fit = fitchange = fitold = f = 0
	grpSize = math.ceil(F.shape[1]/numGroups)
	# pdb.set_trace()
	numRows = F.shape[1]
	# **** Important Note change F to F.T for equiavalence with the old update steps
	# pdb.set_trace()
	for itr in range(maxIter):
		fitold = fit
		tic = time.time()
		#Check F shape
		if groupSparseF:
			# print("Group Sparse Update of F")
			for grpNum in range(numGroups):
				Dg = [mat[:,int(grpNum*grpSize):min(int((grpNum+1)*grpSize),numRows)] for mat in D]
				Ag = [mat[:,int(grpNum*grpSize):min(int((grpNum+1)*grpSize),numRows)] for mat in A]
				# for i,mat in enumerate(Dg):
				# 	print("Dg[%d] rank : %f\t Dg[%d] shape : %s\n"%(i,np.linalg.matrix_rank(mat),i,mat.shape))

				# for i,mat in enumerate(Ag):
				# 	print("Ag[%d] rank : %f\t Ag[%d] shape : %s\n"%(i,np.linalg.matrix_rank(mat),i,mat.shape))
				LOSSTOTAL, LOSSACT = _compute_RMSETOTAL(D,A,F,P,O)
				# print("Loss during an update of Fg : %f"%LOSSTOTAL)
				# print("Number of non-zeros in F: %d"%sum(np.count_nonzero(F,axis = 0)))
				F[:,int(grpNum*grpSize):int((grpNum+1)*grpSize)] = _updateFGS(Dg,Ag,F[:,int(grpNum*grpSize):min(int((grpNum+1)*grpSize),numRows)],P,O,maxIter,lmbdaF)
			# F[0,:] = np.zeros(numRows)
			# print("*****")
			# check_non_negative(F,'update of F')
			sumRF = np.ndarray.sum(F,axis= 1)[:,np.newaxis]
			F = F/sumRF
		else:
			# print("Multiplicative Update of F")
			F = _updateF(D,A,F,P,O,k,S,spatialGrad,gradNorm,gradOp,lmbdaF,lmbdaTV)
			# pdb.set_trace()
			# print("k = %f \t || S-FF'||**2 = %f"%(k,np.linalg.norm(S-F@F.T)**2))
		for index in range(len(D)):
			P[index] = _updatePEye(D[index],F,P[index],index,lmbdaPi)
			# check_non_negative(P[index], 'update P[%d]'%index)

		# for index in range(len(D)):
		# 	# S[index] = _updateSi(D[index],F.T,P[index],S[index],index,lmbdaSR)
		# 	U[index] = _updateUi(D[index],F,P[index],U[index],sigma[index],V[index],index,lmbdaSR,lmbdaSimplex)
		# 	V[index] = _updateVi(D[index],F,P[index],U[index],sigma[index],V[index],index,lmbdaSR,lmbdaSimplex)
		# 	sigma[index] = _updateSigi(D[index],F,P[index],U[index],sigma[index],V[index],index,lmbdaSR/2,lmbdaSimplex)
		# S = [U[i]@sigma[i]@V[i] for i in range(len(D))]

			# check_non_negative(S[index],'update S[%d]'%index)

		for index in range(len(A)):
			O[index] = _updateOJ(A[index],F,O[index],index,lmbdaOj)
			# check_non_negative(O[index],'update O[%d]'%index)

		# for index in range(len(A)):
		# 	# R[index] = _updateRj(A[index],F.T,O[index],R[index],index,lmbdaSR)
		# 	L[index] = _updateUi(A[index],F,O[index],L[index],T[index],W[index],index,lmbdaSR,lmbdaSimplex)
		# 	W[index] = _updateVi(A[index],F,O[index],L[index],T[index],W[index],index,lmbdaSR,lmbdaSimplex)
		# 	T[index] = _updateSigi(A[index],F,O[index],L[index],T[index],W[index],index,lmbdaSR/2,lmbdaSimplex)
		# 	# check_non_negative(R[index],'update R[%d]'%index)
		# R = [L[i]@T[i]@W[i] for i in range(len(A))]

		if compute_fit:
			LOSSTOTAL, LOSSACT = _compute_RMSETOTAL(D,A,F,P,O)
			fit = LOSSTOTAL
		else:
			fit = itr


		toc = time.time()

		exectime = abs(toc-tic)

		fitchange = abs(fitold - fit)
		if fitold<fit and fitold != 0:
			print("Error Increased")
			# pdb.set_trace()

		print('[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' % (
			itr, fit, fitchange, exectime
		))

		if itr > 0 and fitchange < conv:
			break




	# fitData,fitAct = _compute_fit(D,A,F,P,O)
	LOSSTOTAL, LOSSACT = _compute_RMSETOTAL(D,A,F,P,O)
	return F,P,O,LOSSTOTAL,LOSSACT



def _updateZ(Z,U,lmbda):
	row,col = Z.shape
	i = 0
	for i in range(row):
		vecPlus = U[i,:].clip(min=0)
		normVecPlus = np.linalg.norm(vecPlus)
		if normVecPlus <= lmbda:
			Z[i,:] = np.zeros(col)
			i+=1
		else:
			Z[i,:] = vecPlus*(1-lmbda/normVecPlus)

	# print("Number of Rows turned 0 : %d"%i)
	return Z




def _updateFGS(D,A,F,P,O,iters=50,lmbdaF = 0.1): # This F is transpose of the old one, so need to change accordingly
	# print('Updating F')
	numChannels = len(D)
	numLayers = len(A)
	C = D+A
	LM = P+O
	# CM = S+R
	B = []
	row,col = F.shape
	# Z = np.random.rand(F.shape)
	Ztilda = Z =  F

	tau = 1
	# Y = tau*F+(1-tau)*Ftilda
	for i in range(numChannels+numLayers):
		B.append(LM[i])

	
	sigmaMax = [max(np.linalg.svd(mat.T@mat,full_matrices = False,compute_uv = False)) for mat in B]
	for i in range(iters):
		normOld = np.linalg.norm(Ztilda)
		Y = tau*Z+(1-tau)*Ztilda
		U = Z - sum([(B[i].T@B[i]@Y - B[i].T@C[i])/((numChannels+numLayers)*tau*sigmaMax[i]) for i in range(numChannels+numLayers)])
		Z = _updateZ(Z,U,lmbdaF*sum([1/sig for sig in sigmaMax])/(tau))
		# Z = Z/Z.max(axis =0)
		Ztilda = tau*Z + (1-tau)*Ztilda
		# Ztilda = Ztilda/Ztilda.max(axis =0)
		tau = max((-1+math.sqrt(1+4*tau))/(2*tau), (-1-math.sqrt(1+4*tau))/(2*tau))
		normNew = np.linalg.norm(Ztilda)
		# pdb.set_trace()
		if tau < 0:
			return Ztilda
		# print("Inner Z norm change %f"%abs(normNew-normOld))
		if abs(normNew-normOld) < 1e-2:
			return Ztilda
		else:
			continue
	
	# sumRF = np.ndarray.sum(Ztilda,axis= 1)[:,np.newaxis]
	return Ztilda#/sumRF

def diffR(mat):
	
	ans = np.zeros(mat.shape)

	r,c = mat.shape
	for i in range(r-1):
		ans[i,:] = mat[i+1,:] - mat[i,:]
	return ans


def diffC(mat):
	
	ans = np.zeros(mat.shape)

	r,c = mat.shape
	for i in range(c-1):
		ans[:,i] = mat[:,i+1] - mat[:,i]
	return ans

gradOp = (diffR,diffC)

def spatialGrad(mat,gradOperator=gradOp):
	SG = []
	for diff in gradOperator:
		SG.append(diff(mat))
	return SG

def gradNorm(tupOfDiff):
	norm = 0
	for mat in tupOfDiff:
		norm+= squared_norm(mat)
	return norm**0.5


def diffRLaplace(mat):
	m = np.zeros(mat.shape)
	r,c = mat.shape
	for i in range(r):
		if i == 0:
			m[i,:] = mat[i,:]
		elif i == r-1:
			m[i,:] = -mat[i-1,:]
		else:
			m[i,:] = mat[i,:]-mat[i-1,:]
	return m 



def diffCLaplace(mat):
	m = np.zeros(mat.shape)
	r,c = mat.shape
	for j in range(c):
		if j == 0:
			m[:,j] = mat[:,j]
		elif j == c-1:
			m[:,j] = -mat[:,j-1]
		else:
			m[:,j] = mat[:,j]-mat[:,j-1]
	return m



def laplacian(grad,div,k = 1):
	M = np.zeros(grad[0].shape)
	for i,mat in enumerate(grad):
		M += div[i](mat)
	return M

def _updateF(D,A,F,P,O,k,S,spatialGrad,gradNorm,gradOp = gradOp, lmbdaF = 0.1,lmbdaTV = 0.1):
	# print('Updating F')
	numChannels = len(D)
	numLayers = len(A)
	row,col = F.shape
	ep = 1e-9
	# Computing numerator part 1
	num1 = np.zeros((row,col))
	for i in range(numChannels): #Parallelize this
		num1+=2*P[i].T@D[i]


	num2 = np.zeros((row,col))
	for j in range(numLayers): #Parallelize this
		num2+=2*O[j].T@A[j]

	num3 = lmbdaTV*laplace(F)/gradNorm(spatialGrad(F,gradOp)) # TotalVariation penalty
	numerator = num1+num2+num3


	## clipping as per paper
	numerator[numerator<ep] = ep

	#Computing Denominator

	den1 = np.zeros((row,col))
	for i in range(numChannels): #Parallelize this
		den1+=2*P[i].T@P[i]@F


	den2 = np.zeros((row,col))
	for j in range(numLayers): #Parallelize this
		den2+=2*O[j].T@O[j]@F

	# den3 = 4*k*F@F.T@F
	den3 = k*(S.T@F+S@F)
	denom = den1+den2+den3+2*lmbdaF*np.ones(F.shape) #Adding reularization L1


	#Computing updated F
	mutliplicand = np.divide(numerator,denom.__iadd__(ep)) #adding epsilon for numerical stability

	F=np.multiply(F,mutliplicand)
	### to make norm of rows of F unit L1 ###
	sumRF = np.ndarray.sum(F,axis= 1)[:,np.newaxis]
	# pdb.set_trace()
	return F/sumRF




def _updatePEye(Di,F,Pi,index,lmbdaPi = 0.0001):
	# print('Updating P[%s]'%str(index))
	row,col = Pi.shape
	ep = 1e-9
	numerator = 2*Di@F.T
	denominator = 2*Pi@F@F.T + 2*lmbdaPi*Pi
	mutliplicand = np.divide(numerator,denominator.__iadd__(ep))
	Pi = np.multiply(Pi,mutliplicand)
	# Clipping and clamping pixel values
	Pi[Pi > 1] = 1
	Pi[Pi < 0] = 0
	return Pi


def _updateOJ(Aj,F,Oj,index,lmbdaOj = 0.0001):
	# print('Updating O[%s]'%str(index))
	row,col = Oj.shape
	ep = 1e-9
	numerator = 2*Aj@F.T
	denominator = 2*Oj@F@F.T + 2*lmbdaOj*Oj
	mutliplicand = np.divide(numerator,denominator.__iadd__(ep))
	Oj = np.multiply(Oj,mutliplicand)
	return Oj


def _compute_RMSETOTAL(D,A,F,P,O):
	fD = 0
	numElementsD = 0
	fA = 0
	numElementsA = 0
	for i,Di in enumerate(D):
		PiF = dot(P[i],F)
		fD+=squared_norm(Di -PiF)
		numElementsD += Di.size

	for i,Ai in enumerate(A):
		OiF = dot(O[i],F)
		fA+=squared_norm(Ai -OiF)
		numElementsA += Ai.size

	answer = (fD+fA)/(numElementsD+numElementsA)

	return math.sqrt(answer),math.sqrt(fA)/numElementsA





