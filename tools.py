import os
import torch
import torch.nn as nn
import torch.nn.functional as FU
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib
#matplotlib.use('agg')
#matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
import argparse
import glob
import PIL
from PIL import Image
import pdb
# from calsSC import cals
from cnmf import cnmf
from numpy import dot, zeros, array, eye, kron, prod
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
import math
from sklearn.preprocessing import normalize
#import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import pickle
from torch.utils.data.sampler import SequentialSampler, RandomSampler, SubsetRandomSampler
from collections import Counter
import random
import pdb
from PIL import Image, ImageDraw, ImageFont, ImageColor
# from parafac2ALS import als
from scipy.optimize import curve_fit
import shutil
# Ignore warnings
import warnings
# from torch._six import inf
from bisect import bisect_right
from functools import partial
from scipy import signal,misc
import copy
import operator
import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from numpy import linalg as LA



def heatmap(data, row_labels, col_labels, ax=None,
			cbar_kw={}, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.

	Parameters
	----------
	data
		A 2D numpy array of shape (N, M).
	row_labels
		A list or array of length N with the labels for the rows.
	col_labels
		A list or array of length M with the labels for the columns.
	ax
		A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
		not provided, use current axes or create a new one.  Optional.
	cbar_kw
		A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
		The label for the colorbar.  Optional.
	**kwargs
		All other arguments are forwarded to `imshow`.
	"""

	if not ax:
		ax = plt.gca()

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	# We want to show all ticks...
	ax.set_xticks(np.arange(data.shape[1]))
	ax.set_yticks(np.arange(data.shape[0]))
	# ... and label them with the respective list entries.
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=True, bottom=False,
				   labeltop=True, labelbottom=False)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
			 rotation_mode="anchor")

	# Turn spines off and create white grid.
	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)

	return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
					 textcolors=["black", "white"],
					 threshold=None, **textkw):
	"""
	A function to annotate a heatmap.

	Parameters
	----------
	im
		The AxesImage to be labeled.
	data
		Data used to annotate.  If None, the image's data is used.  Optional.
	valfmt
		The format of the annotations inside the heatmap.  This should either
		use the string format method, e.g. "$ {x:.2f}", or be a
		`matplotlib.ticker.Formatter`.  Optional.
	textcolors
		A list or array of two color specifications.  The first is used for
		values below a threshold, the second for those above.  Optional.
	threshold
		Value in data units according to which the colors from textcolors are
		applied.  If None (the default) uses the middle of the colormap as
		separation.  Optional.
	**kwargs
		All other arguments are forwarded to each call to `text` used to create
		the text labels.
	"""

	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.

	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			  verticalalignment="center")
	kw.update(textkw)

	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			texts.append(text)

	return texts


def HeatMap(mat,numToClassX,numToClassY,title,cbarTitle,saveTo,annotate = False):
	y,x = mat.shape
	numClassesX = x
	numClassesY = y
	if annotate:
		classLabelsX = [numToClassX[i] for i in range(numClassesX)]
		classLabelsY = [numToClassY[i] for i in range(numClassesY)]
	else:
		classLabelsX = []
		classLabelsY = []


	fig, ax = plt.subplots()
	im, cbar = heatmap(mat, classLabelsY, classLabelsX, ax=ax,
					cmap="YlGn", cbarlabel=cbarTitle)
	# if annotate:
	# 	texts = annotate_heatmap(im, valfmt="{x:.1f} t")

	ax.set_title(title)
	fig.tight_layout()
	plt.savefig(saveTo)
	plt.clf()

	


def floatMatrixToGS(matrix,saveAs,magFactorY = 1,magFactorX = 1): # To Visualize P matrix
	""" SaveAs contains the filepath and filename"""
	# max = matrix.max()
	# matrix.__imul__(255/max)
	img = PIL.Image.fromarray(matrix)
	img = img.convert('RGB')
	img = img.resize((magFactorY*matrix.shape[0],magFactorX*matrix.shape[1]))
	img.save(saveAs)



def flattenMatrix(matrix,sortTrue = True):
	''' Returns a list of tuples of the form [(row index,col index, value at the index),] '''
	flattenedMatrix = [] # containing the matrix info
	numRow,numCol = matrix.shape
	for currentRowIndex in range(0,numRow):
		for currentColIndex in range(0,numCol):
			flattenedMatrix.append((currentRowIndex,currentColIndex,matrix[currentRowIndex][currentColIndex]))
	if sortTrue:
		flattenedMatrix.sort(key = operator.itemgetter(2), reverse = True)
	return flattenedMatrix



def anaylzeInputLatentDim(P,S,F,path):
	l = len(S)


def generateLatentImages(P,path):
	l = len(P)
	numImages = P[0].shape[1]
	for imgNum in range(numImages):
		image = []
		for channel in range(l):
			imgC = P[channel][:,imgNum].reshape(32,32)
			# pdb.set_trace()
			imgC.__imul__(255/imgC.max())
			# imgC[imgC > 64] = 1
			# imgC[imgC <= 64] = 0
			# imgC.__imul__(255)
			# pdb.set_trace()
			image.append(np.uint8(imgC))
		if l != 1:
			image = tuple(image)
			M = np.dstack(image)
		elif l == 1:
			M = image[l-1]
		# pdb.set_trace()
		floatMatrixToGS(M,os.path.join(path,'Latent Factor: '+str(imgNum)+'.jpg')) # removing magnification


def floatMatrixToHeatMapSNS(matrix,saveAs,title = "", xLabel= "", yLabel = "", dpi = 1200,cmap = 'hot', annot = False,linewidths = 0,fmt="f",annotFontSize = 10):#Pyplot
	ax = sns.heatmap(matrix,cmap = cmap, annot = annot,annot_kws={"size": annotFontSize},linewidths = linewidths,fmt=fmt)
	plt.title(title, fontsize = 20) 
	plt.xlabel(xLabel, fontsize = 15) 
	plt.ylabel(yLabel, fontsize = 15)

	plt.savefig(saveAs,dpi = dpi)
	plt.clf()

def cosineSimilarity(mat, rowMode = True):
	''' returns cosine similarity between rows of a matrix'''
	if rowMode==False:
		mat = mat.T

	rowNorm = np.reshape(np.linalg.norm(mat,axis=1),(mat.shape[0],1))
	outerProdRowNorms = rowNorm@rowNorm.T

	return mat@mat.T/outerProdRowNorms


def mutualCoherence(mat,rowMode = True):
	''' returns mutual Coherence between rows of a matrix'''
	if rowMode==False:
		mat = mat.T

	rowNorm = np.reshape(np.linalg.norm(mat,axis=1),(mat.shape[0],1))
	# outerProdRowNorms = rowNorm@rowNorm.T
	M = mat/rowNorm
	MMT = M@M.T
	np.fill_diagonal(MMT,0)


	return (MMT.max(),MMT.mean(),MMT.min())


def generateLatentActivations(O,path , title1 = "",  title2 = "", title3 = "", xLabel1= "", yLabel1 = "",  xLabel2= "", yLabel2 = "",  xLabel3= "", yLabel3 = "",):
	numLayers = len(O)
	numLatentImages,numDim = O[0].shape
	fontSize = 10
	annotations = True
	if numDim>100:
		annotations = False
	elif numDim>10:
		fontSize *= 10/numDim
	for layer in range(numLayers):
		suffix1 = os.path.join(path,'LayerNum%d.jpg'%layer )
		# suffix2 = os.path.join(path,'LayerLatentDimSimilarity%d.jpg'%layer )
		suffix3 = os.path.join(path,'LayerLatentCosineSimilarity%d.jpg'%layer )
		floatMatrixToHeatMapSNS(O[layer],suffix1,  title1 , xLabel1, yLabel1, annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)
		# floatMatrixToHeatMapSNS(O[layer].T@O[layer],suffix2)
		floatMatrixToHeatMapSNS(cosineSimilarity(O[layer].T), suffix3, title3 , xLabel3, yLabel3, annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)

def flattenSortedMatrix(matrix,t=0):
		''' Returns a list of tuples of the form [(row index,col index, value at the index),] '''
		flattenedMatrix = [] # containing the matrix info
		numRow,numCol = matrix.shape
		norm = np.linalg.norm(matrix)
		for currentRowIndex in range(0,numRow):
			for currentColIndex in range(0,numCol):
				value = matrix[currentRowIndex][currentColIndex]/norm
				if value >= t:
					flattenedMatrix.append((currentRowIndex,currentColIndex,value))
		flattenedMatrix.sort(key = lambda x:x[2], reverse = True)
		return flattenedMatrix


def matToBPG(matrix,path):
	row,col = matrix.shape
	str1 = 'RLD'
	str2 = 'CLD'
	m1 = [str1+str(i)for i in range(row)] # bipartite set 1
	m2 = [str2+str(i)for i in range(col)] # bipartite set 2
	flatMatrix = flattenSortedMatrix(matrix,0.1) #edges
	edgeList = [(m1[tup[0]],m2[tup[1]]) for tup in flatMatrix]
	pos = {}
	pos.update((node,(1,index)) for index,node in enumerate(m1))
	pos.update((node,(2,index)) for index,node in enumerate(m2))
	B = nx.Graph()
	B.add_nodes_from(m1,bipartite=0)
	B.add_nodes_from(m2,bipartite=1)
	B.add_edges_from(edgeList)
	nx.draw_networkx(B,pos = pos)
	plt.savefig(path,dpi = 2000)
	plt.clf()
	P = bipartite.projected_graph(B,set(m2))
	Q = bipartite.projected_graph(B,set(m1))
	pos1 = nx.circular_layout(P)
	nx.draw_networkx(P,pos = pos1)
	plt.savefig(path+'ClassNodeProjects',dpi = 2000)
	plt.clf()


def matToText(matrix,path,LoD,numToClass):
	flatMatrix = flattenSortedMatrix(matrix)
	dictData = {}
	for tup in flatMatrix:
		if tup[0] not in dictData:
			dictData[tup[0]] = [(tup[1],tup[2])]
		else:
			dictData[tup[0]].append((tup[1],tup[2]))

	with open(path,'w') as fH:
		for key in dictData:
			fH.write('*****Left Latent Space Dimension: %d *****\n'%(key))
			for tup in dictData[key]:
				fH.write('Class Latent Space Dimension: %d\tValue: %f\n'%(tup))
				# pdb.set_trace()
				fH.write(dictToLine(LoD[tup[0]],numToClass)+'\n')
			fH.write('\n\n')

	fH.close()


def spaceSimilarity(LoM,path,LoD,numToClass):

	for i,matrix in enumerate(LoM):
		fname = 'matrix%d'%i
		Gfname = 'Graphmatrix%d'%i
		textFname = 'TextFileName%d'%i
		# rowMaxes = np.amax(matrix,axis = 0)
		# matrixTilda = np.true_divide(matrix,rowMaxes)

		matToText(matrix,os.path.join(path,textFname),LoD,numToClass)
		maximum = matrix.max()
		matrix /= maximum
		matToBPG(matrix,os.path.join(path,Gfname))
		# matrix.__div__(maximum)
		matrix.__imul__(255.999)
		floatMatrixToHeatMapSNS(np.uint8(matrix),os.path.join(path,fname))


	


def analyzeF(F,labelList):
	vF = np.flipud(np.sort(F,axis = 0,kind = 'mergesort'))
	iF = np.flipud(np.argsort(F,axis = 0,kind = 'mergesort'))
	lF = np.zeros(iF.shape)
	for index,val in np.ndenumerate(iF):
		# pdb.set_trace()
		lF[index] = labelList[int(val)]

	return vF,iF,lF




# def generateReportF(vF,lF):
# 	''' squared norm unity'''
# 	LoD = []
# 	r,c = vF.shape
# 	for j in range(c):
# 		sqNormj = squared_norm(vF[:,j])
# 		d = {}
# 		for i in range(r):
# 			if int(lF[i,j]) in d:
# 				d[int(lF[i,j])] += (vF[i,j])**2/sqNormj
# 			else:
# 				d[int(lF[i,j])] = vF[i,j]**2/sqNormj
# 		LoD.append(d)
# 	return LoD


def generateReportF(vF,lF):
	''' L1 row norm unity or column, depending on iF F.T was passed to the function'''
	LoD = []
	r,c = vF.shape
	for j in range(c):
		sqNormj = sum(vF[:,j])
		d = {}
		for i in range(r):
			if int(lF[i,j]) in d:
				d[int(lF[i,j])] += (vF[i,j])/sqNormj
			else:
				d[int(lF[i,j])] = vF[i,j]/sqNormj
		LoD.append(d)
	return LoD


def dictToLine(d,numToClass):
	revSortedD = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
	line = ''
	entropy = 0
	for classExample,weight in revSortedD:
		# pdb.set_trace()
		line  += 'Class : %s - Weight : %f\t'%(str(numToClass.get(classExample)),weight)
		if weight != 0:
			entropy -= weight*np.log(weight)
	return 'Entropy : '+str(entropy)+'\t'+line.strip()+'\n'


def FReport(LoD,saveTo,numToClass,fname = 'FReport.txt'):
	with open(os.path.join(saveTo,fname),'w') as fH:
		for i,d in enumerate(LoD):
			pre = 'Latent Factor : %d\t'%i
			line = dictToLine(d,numToClass)
			fH.write(pre+line)



def plotLists(lol,saveTo,legends = ['max','mean','min']):
	numLists = len(lol)
	plt.figure()
	for l in lol:
		plt.plot(l)
	plt.legend(legends,loc = 'upper right')
	plt.savefig(saveTo)
	plt.clf()



def vFlFToClassMat(vF,lF):
	''' transpose taken because F.T was used while calculating vF,lF'''
	numClasses = len(Counter(lF[:,0]))
	vF,lF = vF.T,lF.T
	M = np.zeros((lF.shape[0],numClasses))
	for rowNum,row in enumerate(lF):
		for colNum,ClassNum in enumerate(row):
			M[rowNum,int(ClassNum)] += vF[rowNum,colNum]

	return M

def pairwiseHellinger(M,rowMode=True):
	if rowMode == False:
		M = M.T
	sqrtM = np.sqrt(M)
	r,c = M.shape
	H = np.zeros((r,r))
	for i in range(r):
		for j in range(r):
			H[i,j] = H[j,i] = (1/np.sqrt(2))*np.sum((sqrtM[i,:]-sqrtM[j,:])**2)**0.5

	return H

def pairwiseBhattacharyya(M,rowMode=True):
	if rowMode == False:
		M = M.T
	sqrtM = np.sqrt(M)
	r,c = M.shape
	H = np.zeros((r,r))
	for i in range(r):
		for j in range(r):
			H[i,j] = H[j,i] = np.sum((sqrtM[i,:]*sqrtM[j,:]))

	return H

def pairwiseKL(M,rowMode=True):
	if rowMode == False:
		M = M.T
	sqrtM = np.sqrt(M)
	r,c = M.shape
	KL = np.zeros((r,r))
	for i in range(r):
		for j in range(r):
			KL[i,j] = np.sum(M[i,:]*np.log((M[i,:]+1e-9)/(M[j,:]+1e-9)))
			KL[j,i] = np.sum(M[j,:]*np.log((M[j,:]+1e-9)/(M[i,:]+1e-9)))
	return KL


def topImagesPerLatentFactor(vF,iF,indexToImagePath,saveTo):
	top = min(200,int(iF.shape[0]/10))
	for LFNUM in range(iF.shape[1]):
		if not os.path.exists(os.path.join(saveTo,'LatentFactor-%d'%LFNUM)):
			os.makedirs(os.path.join(saveTo,'LatentFactor-%d'%LFNUM))
		sumFactor = sum(vF[:,LFNUM])
		weightCollected = sum(vF[:top,LFNUM])
		for imageNum,imageIndex in enumerate(iF[:top,LFNUM]):
			imagePath = indexToImagePath[imageIndex]
			fname = imagePath.split('/')[-3].strip()+'_'+imagePath.split('/')[-2].strip()+'_'+imagePath.split('/')[-1].strip().split('.')[0]
			fname = str(imageNum+1)+'_'+fname+'_'+'Weight : %s'%str(vF[imageNum,LFNUM])+'_'+'Per : %s'%str(100*vF[imageNum,LFNUM]/sumFactor)+'_'+'Top_Weight : %s'+str(weightCollected)
			shutil.copy2(imagePath,os.path.join(saveTo,'LatentFactor-%d'%LFNUM,fname+'.png'))


def topMaskedImagesPerLatentFactor(vF,iF,P,indexToImagePath,saveTo):
	top = min(200,int(iF.shape[0]/10))
	numChannels = len(P)
	for LFNUM in range(iF.shape[1]):
		if not os.path.exists(os.path.join(saveTo,'LatentFactor-%d'%LFNUM)):
			os.makedirs(os.path.join(saveTo,'LatentFactor-%d'%LFNUM))
		sumFactor = sum(vF[:,LFNUM])
		weightCollected = sum(vF[:top,LFNUM])
		for imageNum,imageIndex in enumerate(iF[:top,LFNUM]):
			imagePath = indexToImagePath[imageIndex]
			img = Image.open(imagePath).convert('RGB')
			# imgArr = np.array(img)
			imgc0 = np.array(img.getchannel(0))
			imgc1 = np.array(img.getchannel(1))
			imgc2 = np.array(img.getchannel(2))

			fname = imagePath.split('/')[-3].strip()+'_'+imagePath.split('/')[-2].strip()+'_'+imagePath.split('/')[-1].strip().split('.')[0]
			fname = str(imageNum+1)+'_'+fname+'_'+'Weight : %s'%str(vF[imageNum,LFNUM])+'_'+'Per : %s'%str(100*vF[imageNum,LFNUM]/sumFactor)+'_'+'Top_Weight : %s'+str(weightCollected)
			
			latentImage = []
			maskedImage = []
			for c in range(numChannels):
				imgC = P[c][:,LFNUM].reshape(32,32)
				imgC.__imul__(255/imgC.max())
				latentImage.append(np.uint8(imgC))

				# binaryImgC[]

			if numChannels != 1:
				latentImage = tuple(latentImage)
				M = np.dstack(latentImage)
				latentFactorImg = PIL.Image.fromarray(M)
				latentFactorImg = latentFactorImg.convert('RGB')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				# pdb.set_trace()
				# latentFactorImgArr[latentFactorImgArr <= 10] = 0
				# imgc0[latentFactorImgArr == 0] = 0
				# imgc1[latentFactorImgArr == 0] = 0
				# imgc2[latentFactorImgArr == 0] = 0
				imgc0[latentFactorImgArr[:,:,0] <= np.median(latentFactorImgArr[:,:,0])] = 0
				imgc1[latentFactorImgArr[:,:,1] <= np.median(latentFactorImgArr[:,:,1])] = 0
				imgc2[latentFactorImgArr[:,:,2] <= np.median(latentFactorImgArr[:,:,2])] = 0

			elif numChannels == 1:
				M = latentImage[l-1]
				latentFactorImg = PIL.Image.fromarray(M)
				latentFactorImg = latentFactorImg.convert('L')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				latentFactorImgArr[latentFactorImgArr <= 10] = 0

			# targetImage = np.multiply(imgArr,M)

			targetImagec0 = imgc0
			targetImagec1 = imgc1
			targetImagec2 = imgc2

			# pdb.set_trace()


			# targetImagec0 = np.multiply(imgc0,latentFactorImgArr)
			# targetImagec1 = np.multiply(imgc1,latentFactorImgArr)
			# targetImagec2 = np.multiply(imgc2,latentFactorImgArr)


			imgMasked = Image.fromarray(np.uint8(np.dstack((targetImagec0,targetImagec1,targetImagec2))))

			imgMasked.save(os.path.join(saveTo,'LatentFactor-%d'%LFNUM,fname+'.png'))




def numToClassText(numToClassDict,saveTo):
	with open(saveTo,'w') as fH:
		for i in range(len(numToClassDict)):
			fH.write(numToClassDict[i]+'\n')



def dictToLineAdv(d,numToClass):
	# revSortedD = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
	purityDict = cleanVsAdvWeight(d,numToClass)
	line = ''
	entropy = 0
	for classExample,weight in purityDict.items():
		# pdb.set_trace()
		line  += 'Class : %s - Weight : %f\t'%(classExample,weight)


	return line.strip()+'\n'

def cleanVsAdvWeight(d,numToClass):
	pureTaintedDict = {'pure': 0, 'adversarial': 0}
	for classExample,weight in d.items():
		if 'Adv' in numToClass.get(classExample):
			pureTaintedDict['adversarial'] += weight
		else:
			pureTaintedDict['pure'] += weight

	return pureTaintedDict




def FAdvReport(LoD,saveTo,numToClass,fname = 'AdvReport.txt'):
	with open(os.path.join(saveTo,fname),'w') as fH:
		for i,d in enumerate(LoD):
			pre = 'Latent Factor : %d\t'%i
			line = dictToLineAdv(d,numToClass)
			fH.write(pre+line)


def dictToLineCutOff(d,numToClass):
	revSortedD = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
	line = ''
	numElem = len(revSortedD)
	for classExample,weight in revSortedD:
		# pdb.set_trace()
		if weight >= 1/numElem:
			line  += 'Class : %s - Weight : %f\t'%(str(numToClass.get(classExample)),weight)
	return line.strip()+'\n'


def FReportCutOff(LoD,saveTo,numToClass,fname = 'FReport-CutOff.txt'):
	with open(os.path.join(saveTo,fname),'w') as fH:
		for i,d in enumerate(LoD):
			pre = 'Latent Factor : %d\t'%i
			line = dictToLineCutOff(d,numToClass)
			fH.write(pre+line)


def dictToLineSameness(d,numToClass):
	revSortedD = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
	# line = ''
	classList = []
	AdvClassList = []
	numElem = len(revSortedD)
	hyphen = "-"
	for classExample,weight in revSortedD:
		# pdb.set_trace()
		if weight >= 1/numElem:
			classString = str(numToClass.get(classExample))
			if hyphen in classString:
				AdvClassList.append(classString[classString.find(hyphen)+1:])
			else:
				classList.append(classString)
	
	classListSet = set(classList)
	AdvClassListSet = set(AdvClassList)

	commons = len(classListSet.intersection(AdvClassListSet))
	# pdb.set_trace()

	return str(commons)+'\n'


def FReportSameness(LoD,saveTo,numToClass,fname = 'FReport-Commons.txt'):
	with open(os.path.join(saveTo,fname),'w') as fH:
		total = 0
		for i,d in enumerate(LoD):
			pre = 'Latent Factor : %d\t'%i
			line = dictToLineSameness(d,numToClass)
			total += int(line.strip())
			fH.write(pre+line)
		fH.write("Total :\t"+str(total))


def analyzeFKNN(F,labelList):
	r,c = F.shape
	numClasses = len(set(labelList))
	numNeighours = 0
	if r > c :
		print("Tall and Thin Matrix")
		numNeighours = int(r/numClasses)+1
	else:
		print("Short and Wide: Switching")
		numNeighours = int(c/numClasses)+1
		F = F.T

	print("*** F Shape *** : %s"%(F.shape,))
	
	tallyMat = np.zeros(F.shape)

	nbrs = NearestNeighbors(n_neighbors=numNeighours, algorithm='auto').fit(F)
	dist , ind = nbrs.kneighbors(F)
	KNG = nbrs.kneighbors_graph(F).toarray()
	##### Removing self links #####
	numExamples = KNG.shape[0]
	for i in range(numExamples):
		KNG[i,i] = 0

	# pdb.set_trace()
	for exampleIndex in range(numExamples):
		# print("Example : %s"%exampleIndex)
		for neighbourIndex in np.nonzero(KNG[exampleIndex,:])[0]:
			# print("\tneighbourIndex : %s"%neighbourIndex)
			maxLatentFactorIndex = np.argmax(F[neighbourIndex,:])
			# print("\tmaxLatentFactorIndex : %s"%maxLatentFactorIndex)
			# pdb.set_trace()
			tallyMat[exampleIndex,maxLatentFactorIndex] += 1

	
	return tallyMat, entropy(tallyMat/tallyMat.sum(axis = 1, keepdims = True), axis = 1, base = 2)


def barPlot(yVals,xVals,title,yLabel,xLabel,saveTo):
	fig = plt.figure()
	# ax = fig.add_axes([0,0,1,1])
	plt.style.use('ggplot')
	plt.rcParams['axes.facecolor'] = 'white'
	plt.gca().set_frame_on(True)
	xpos = [i for i in range(len(xVals))]
	plt.title(title, fontsize=22)
	plt.ylabel(yLabel,fontsize=18)
	plt.xlabel(xLabel,fontsize=18)
	plt.bar(xpos,yVals,color='green')
	plt.xticks(xpos, xVals,rotation = 90)
	plt.tight_layout()
	# plt.figure(facecolor="white")
	plt.savefig(saveTo)
	plt.clf()

def analyzeFKNNPlots1(distMat,entArr,num2ClassDict,num2SuperClassDict,classLabelList,superClassLabelList,pathList,saveTo):
	if not os.path.exists(saveTo):
		os.makedirs(saveTo)
		os.makedirs(os.path.join(saveTo,'perExamplePlots'))


	revSortedEntArrInd = np.flipud(np.argsort(entArr))
	# pdb.set_trace()
	classLabelListRS = [classLabelList[i] for i in revSortedEntArrInd]
	superClassLabelListRS = [superClassLabelList[i] for i in revSortedEntArrInd]
	classLabelListRSTrueLabel = [num2ClassDict[classLabel] for classLabel in classLabelListRS]
	superClassLabelListRSTrueLabel = [num2SuperClassDict[sclassLabel] for sclassLabel in superClassLabelListRS]

	pathListRS = [pathList[i] for i in revSortedEntArrInd]
	# pdb.set_trace()
	with open(os.path.join(saveTo,'ImagePaths.txt'),'w') as fH:
		for i,path in enumerate(pathListRS):
			fH.write('ImgNum_%s\tEntropy_%s\tClass_%s\tSuperClass_%s\t%s\n'%(i,revSortedEntArrInd[i],classLabelListRSTrueLabel[i],superClassLabelListRSTrueLabel[i],path))
			barPlot(distMat[revSortedEntArrInd[i],:],["LF-%s"%i for i in range(len(distMat[revSortedEntArrInd[i],:]))],"Class-%s"%classLabelListRSTrueLabel[i],"#Nearest Neighbours", "Discovered Concepts", os.path.join(saveTo,'perExamplePlots','ImgNum_%s.png'%i) )
	

def analyzeFKNNPlots2(distMat,entArr,num2ClassDict,num2SuperClassDict,classLabelList,superClassLabelList,pathList,predList,saveTo):
	if not os.path.exists(saveTo):
		os.makedirs(saveTo)
		os.makedirs(os.path.join(saveTo,'perExamplePlots'))

	sortedArrInd = np.argsort(entArr)
	revSortedEntArrInd = np.flipud(sortedArrInd)
	# pdb.set_trace()
	classLabelListRS = [classLabelList[i] for i in revSortedEntArrInd]
	classLabelListS = [classLabelList[i] for i in sortedArrInd]
	superClassLabelListRS = [superClassLabelList[i] for i in revSortedEntArrInd]
	predListRS = [predList[i] for i in revSortedEntArrInd]
	predListS = [predList[i] for i in sortedArrInd]
	classLabelListRSTrueLabel = [num2ClassDict[classLabel] for classLabel in classLabelListRS]
	predListRSTrueLabel = [num2ClassDict[predclassLabel] for predclassLabel in predListRS]
	superClassLabelListRSTrueLabel = [num2SuperClassDict[sclassLabel] for sclassLabel in superClassLabelListRS]

	accVsEntListBool = [classLabelListS[i] == predListS[i] for i in range(len(predListS))]
	accVsEntList = [100*sum(accVsEntListBool[:i+1])/(i+1) for i in range(len(accVsEntListBool))]


	pathListRS = [pathList[i] for i in revSortedEntArrInd]
	# pdb.set_trace()
	fig = plt.figure()
	plt.style.use('ggplot')
	plt.rcParams['axes.facecolor'] = 'white'
	plt.gca().set_frame_on(True)
	plt.title("Accuracy vs Impurity of K-NN", fontsize=20)
	plt.ylabel("Accuracy",fontsize=20)
	plt.xlabel("Impurity of K-NN",fontsize=20)
	plt.plot(accVsEntList, label = "Accuracy")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(saveTo,"AccvsImpurity.png"))
	plt.clf()
	# pdb.set_trace()
	with open(os.path.join(saveTo,'ImagePathsPredLabel.txt'),'w') as fH:
		for i,path in enumerate(pathListRS):
			fH.write('ImgNum_%s\tEntropy_%s\tClass_%s\tSuperClass_%s\tPredictedClass_%s\t%s\n'%(i,revSortedEntArrInd[i],classLabelListRSTrueLabel[i],superClassLabelListRSTrueLabel[i],predListRSTrueLabel[i],path))
			# barPlot(distMat[revSortedEntArrInd[i],:],["LF-%s"%i for i in range(len(distMat[revSortedEntArrInd[i],:]))],"Class-%s"%classLabelListRSTrueLabel[i],"#Nearest Neighbours", "Discovered Concepts", os.path.join(saveTo,'perExamplePlots','ImgNum_%s.png'%i) )
	

	
def neuralEvalAnalysis(OList,saveTo, numEvals=7):
	print("*** Plotting Eigenvalues ***")

	EValList = np.zeros((len(OList),numEvals))
	EValFull = np.zeros((len(OList),OList[0].shape[1]))
	for j,Oj in enumerate(OList):
		EValList[j,:] = LA.eigvals(cosineSimilarity(Oj,False))[:numEvals]
		EValFull[j,:] = LA.eigvals(cosineSimilarity(Oj,False))

	EVALMeanP = np.mean(EValList, axis = 1)
	EValList = EValList.T
	# EValFull = EValFull.T
	EVALMean = np.mean(EValFull, axis = 1)
	

	numLayers = len(OList)
	xLabel = ["Layer #%s"%i for i in range(numLayers)]
	xpos = [i for i in range(len(xLabel))]

	fig = plt.figure()
	plt.style.use('ggplot')
	plt.rcParams['axes.facecolor'] = 'white'
	plt.gca().set_frame_on(True)
	plt.title("Mean of First %s Eigenvalues of Cosine Similarity of Neurons vs Layers"%numEvals, fontsize=14)
	plt.ylabel("Eigenvalue",fontsize=14)
	plt.xlabel("Layer Number",fontsize=14)

	# for e in range(numEvals):
	# 	plt.plot(xpos, EValList[e,:], label = "Lambda %s"%(e+1,))
	plt.plot(xpos, EValList[0,:], label = "Lambda 1")
	# plt.plot(xpos, EVALMean, label = "Lambda : Mean")
	plt.plot(xpos, EVALMeanP, label = "Lambda : Mean:First-%s"%numEvals)

	plt.xticks(xpos, xLabel,rotation = 90)

	plt.legend()
	plt.tight_layout()
	plt.savefig(saveTo)
	plt.clf()



