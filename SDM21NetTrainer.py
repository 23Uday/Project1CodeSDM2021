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
import scipy as sp
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
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.sparse import csgraph
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
# from SSC1 import ADMMLASSOSSC, affinityMatrix
# from graphAnalysis import AffMatClusteringCoeff, matClusteringCoeff, mat2KnnGraph, plotModAcc

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser() # argparser object
parser.add_argument("rootDir", help = "Enter the name of root folder which containts the data subfolders :",type = str)
parser.add_argument("rootDirTest", help = "Enter the name of root folder which containts the test data subfolders :",type = str)
parser.add_argument("outputFolderName", help = "Enter the name(Path) of the Output Folder :", type = str)
parser.add_argument("networkType", help = "Enter the architecture of the network :", type = str)
parser.add_argument("NetworkName", help = "Enter the name(Path) of the network file :", type = str)
# parser.add_argument("Affinity", help = "Enter Affinity matrix type :", type = str, choices = ('knn', 'ssc'))
# parser.add_argument("Rank2", help = "Enter Rank 2 :", type = int)
# parser.add_argument("Rank3", help = "Enter Rank 2 :", type = int)
# parser.add_argument("numGroups", help = "Enter the number of groups: ", type = int)
# parser.add_argument("maxIters", help = "Enter Max no. of iterations: ", type = int,default = 100)
# parser.add_argument("tau", help = "tau ADMM ", type = float, default = 1)
# parser.add_argument("mu", help = "mu ADMM ", type = float, default = 10)
# parser.add_argument("err", help = "error tolerance ADMM ", type = float, default = 1e-2, choices  = (0.1, 0.01, 0.001, 0.0001, 0.00001))
# parser.add_argument("lmbdaSR", help = "Enter lambda S or R: ", type = float)
# parser.add_argument("lmbdaSimplex", help = "Enter lambda Simplex :", type = float)
# parser.add_argument("lmbdaF", help = "Enter lambda F: ", type = float)
# parser.add_argument("lmbdaTV", help = "Enter lambda TV: ", type = float)
# parser.add_argument("k", help = "Enter orthogonality penalty: ", type = float)
parser.add_argument("samplingFactor", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("samplingFactorTest", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("lr", help = "learning rate for SGD to be used: ", type = float)
parser.add_argument("wd", help = "Enter the weight decay: ", type = float)
parser.add_argument("numEpochs", help = "Enter the number of epochs: ", type = int)
parser.add_argument('opt',help = "Enter the optimization algorithm", type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
# parser.add_argument('P_init',help = "Enter the initialization for P", type=str, default='random', choices=('random', 'ID', 'NNDSVD'))
parser.add_argument("classBased", help = "Enter if class based training is required ", type = str, default = 'False',choices = ('True','False'))
# parser.add_argument("classListTrain", help = "Enter the list of classes, eg [1,2,3,4]: ",type = str,default = "['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
# parser.add_argument("classListTest", help = "Enter the list of classes, eg [1,2,3,4]: ",type = str,default = "['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
parser.add_argument("actOnlyMode", help = "Enter true if Data matrices have to be ignored", type = str, default = 'False',choices = ('True','False'))
# parser.add_argument("groupSparseF", help = "Enter true if F matrix is to have group sparsity", type = str, default = 'False',choices = ('True','False'))


args = parser.parse_args()


rootDir = args.rootDir
rootDirTest = args.rootDirTest

outputFolderName = args.outputFolderName
if outputFolderName == 'pwd':
	outputFolderName = os.getcwd()

networkType = args.networkType
NetworkName = args.NetworkName

# Affinity = args.Affinity
# Rank2 = args.Rank2
# Rank3 = args.Rank3
# numGroups = args.numGroups
# maxIters = args.maxIters
# tau = args.tau
# mu = args.mu
# err = args.err
# lmbdaSR = args.lmbdaSR
# lmbdaSimplex = args.lmbdaSimplex
# lmbdaF = args.lmbdaF
# lmbdaTV = args.lmbdaTV
# k = args.k
samplingFactor = args.samplingFactor
samplingFactorTest = args.samplingFactorTest
if samplingFactor > 1:
	samplingFactor = 1.0
elif samplingFactor < 0:
	samplingFactor = 1.0


if samplingFactorTest > 1:
	samplingFactorTest = 1.0
elif samplingFactorTest < 0:
	samplingFactorTest = 1.0


lr = args.lr

wd = args.wd
numEpochs = args.numEpochs
opt = args.opt
# P_init=args.P_init
classBased = args.classBased
# classListTrainString = """[0,1,2,3,4,5,6,7,8,9]"""
# classListTrainString = """[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck]"""
classListTrainString = """[beaver, dolphin, otter, seal, whale, 
aquarium_fish, flatfish, ray, shark, trout, 
orchid, poppy, rose, sunflower, tulip, 
bottle, bowl, can, cup, plate, 
apple, mushroom, orange, pear, sweet_pepper, 
clock, keyboard, lamp, telephone, television, 
bed, chair, couch, table, wardrobe, 
bee, beetle, butterfly, caterpillar, cockroach, 
bear, leopard, lion, tiger, wolf, 
bridge, castle, house, road, skyscraper, 
cloud, forest, mountain, plain, sea, 
camel, cattle, chimpanzee, elephant, kangaroo, 
fox, porcupine, possum, raccoon, skunk, 
crab, lobster, snail, spider, worm, 
baby, boy, girl, man, woman, 
crocodile, dinosaur, lizard, snake, turtle, 
hamster, mouse, rabbit, shrew, squirrel, 
maple_tree, oak_tree, palm_tree, pine_tree, willow_tree, 
bicycle, bus, motorcycle, pickup_truck, train, 
lawn_mower, rocket, streetcar, tank, tractor]"""
# args.classListTrain

# classListTestString = """[0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]"""
# classListTestString = """[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck]"""
classListTestString = """[beaver, dolphin, otter, seal, whale, 
aquarium_fish, flatfish, ray, shark, trout, 
orchid, poppy, rose, sunflower, tulip, 
bottle, bowl, can, cup, plate, 
apple, mushroom, orange, pear, sweet_pepper, 
clock, keyboard, lamp, telephone, television, 
bed, chair, couch, table, wardrobe, 
bee, beetle, butterfly, caterpillar, cockroach, 
bear, leopard, lion, tiger, wolf, 
bridge, castle, house, road, skyscraper, 
cloud, forest, mountain, plain, sea, 
camel, cattle, chimpanzee, elephant, kangaroo, 
fox, porcupine, possum, raccoon, skunk, 
crab, lobster, snail, spider, worm, 
baby, boy, girl, man, woman, 
crocodile, dinosaur, lizard, snake, turtle, 
hamster, mouse, rabbit, shrew, squirrel, 
maple_tree, oak_tree, palm_tree, pine_tree, willow_tree, 
bicycle, bus, motorcycle, pickup_truck, train, 
lawn_mower, rocket, streetcar, tank, tractor]"""
#args.classListTest
actOnlyMode = args.actOnlyMode
# groupSparseF = args.groupSparseF
# if groupSparseF == 'True':
# 	groupSparseF = True
# elif groupSparseF == 'False':
# 	groupSparseF = False


classListTrain = set([str(x.strip()) for x in classListTrainString.strip('[]').strip().split(',')]) # Make set for efficiency
classListTest = set([str(x.strip()) for x in classListTestString.strip('[]').strip().split(',')]) # Make set for efficiency

samplingAdd = 'samplingFactor'+str(samplingFactor)

# outputFolderName =  outputFolderName+'samplingFactor'+str(samplingFactor)
outputFolderName = outputFolderName+"lr:%s-wd:%s-TestFactor:%s"%(lr,wd,samplingFactorTest)
if classBased == 'True':
	# outputFolderName+='-Train-'+str(classListTrainString)+'-Test-'+classListTestString+'-'
	outputFolderName+='-Train-'+str(len(classListTrain))+'-Test-'+str(len(classListTest))+'-'
else:
	outputFolderName+='-sampling-'
if actOnlyMode == 'True':
	outputFolderName += 'ActivationOnly'
else:
	outputFolderName += 'Full'
try:
	os.makedirs(outputFolderName)
except:
	pass

cwd = os.getcwd()

train_batch_size = 64
test_batch_size = 1


### For testing data loader
# dataSet = '/home/usain001/research/DataSets/CifarLocal/train/'

class Dataset(Dataset):
	""" CIFAR 10 dataset """
	def __init__(self,root_dir,all_image_paths = [], all_labels = [],transform =None, sampleSize = 1.0):
		"""
		root_dir : The testing/training data set which contains subfolders.
		sampleSize needs to be between 0 and 1
		"""

		#Raw Data
		self.root_dir = root_dir
		self.transform = transform
		self.classToNum = {}
		# self.superClassToClassNum = {}
		self.classToSuperClassMap = {}
		self.superClassSet = {}
		self.all_labels = []
		self.all_Superlabels = []
		self.all_image_paths = glob.glob(root_dir+"/**/*.png", recursive=True)
		self.all_Classlabels = [fadd.split('/')[-2] for fadd in self.all_image_paths ]
		self.all_SuperClasslabels = [fadd.split('/')[-3] for fadd in self.all_image_paths ]
		# pdb.set_trace()
		for fadd in self.all_image_paths:
			if fadd.split('/')[-2] not in self.classToNum:
				self.classToNum.update({fadd.split('/')[-2]: len(self.classToNum)})
				# self.superClassToClassNum.update({fadd.split('/')[-3]: len(self.classToNum)})
				if fadd.split('/')[-3] not in self.superClassSet:
					self.superClassSet.update({fadd.split('/')[-3]:len(self.superClassSet)})

				### mapping the class to super class
				self.classToSuperClassMap.update({self.classToNum[fadd.split('/')[-2]]:self.superClassSet[fadd.split('/')[-3]] })


				self.all_labels.append(self.classToNum[fadd.split('/')[-2]])
				self.all_Superlabels.append(self.superClassSet[fadd.split('/')[-3]])
			else:
				self.all_labels.append(self.classToNum[fadd.split('/')[-2]])


				if fadd.split('/')[-3] not in self.superClassSet:
					self.superClassSet.update({fadd.split('/')[-3]:len(self.superClassSet)})


				self.all_Superlabels.append(self.superClassSet[fadd.split('/')[-3]])


		self.numToClass = {v: k for k, v in self.classToNum.items()}
		self.superClassSetReverse = {v: k for k, v in self.superClassSet.items()}
		self.classes = set(self.all_labels)

		# preprocessing
		self.counter = Counter(self.all_labels)
		self.sampleSize = sampleSize


		self.LabelToIndex = self.generateLabelsToIndex()
		self.sampledLabelToIndex = self.generateSampledLabelsToIndex()


		# Sampled Data
		self.all_sampled_idx = self.generateSampledIdx() 
		self.all_sampled_labels = [self.all_labels[i] for i in self.all_sampled_idx] # to be used for all labels
		self.all_sampled_super_labels = [self.all_Superlabels[i] for i in self.all_sampled_idx] # to be used for all labels
		self.sampled_counter = Counter(self.all_sampled_labels)
		self.all_sampled_image_paths = [self.all_image_paths[i] for i in self.all_sampled_idx] # Later, add the ability to return this in both classes






	def __len__(self):
		""" returns the total number of files"""
		return len(self.all_sampled_image_paths)


	def __getitem__(self,idx):
		""" return image for the given index"""
		imgAdd = self.all_sampled_image_paths[idx]
		img = Image.open(imgAdd).convert('RGB')
		if self.transform == None:
			img = transforms.ToTensor()(img)
		else:
			img = self.transform(img)
		# pdb.set_trace(0)
		return (img,self.all_sampled_labels[idx],self.all_sampled_super_labels[idx],self.all_sampled_image_paths[idx])


	# Helper Functions	

	def getLabels(self):
		return self.all_labels

	def generateLabelsToIndex(self):
		LabelToIndex = {}
		for idx,label in enumerate(self.all_labels):
			if label not in LabelToIndex:
				LabelToIndex.update({label:[idx]})
			else:
				LabelToIndex[label].append(idx)
		return LabelToIndex


	def getLabelsToIndex(self):
		return self.LabelToIndex

	def generateSampledLabelsToIndex(self):
		sampledLabelToIndex = {}
		for label, idx_list in self.LabelToIndex.items():
			indices = random.sample(range(len(idx_list)), int(len(idx_list)*self.sampleSize))
			sampledLabelToIndex.update({label: [idx_list[i] for i in sorted(indices)]})

		return sampledLabelToIndex


	def generateSampledIdx(self):
		all_sampled_idx = []
		for label,idx_list in self.sampledLabelToIndex.items():
			all_sampled_idx += idx_list
		return all_sampled_idx

	def getInputChannels(self):
		return 3


class DatasetClassBased(Dataset):
	""" Handwritten digits dataset """
	def __init__(self,root_dir,all_image_paths = [], all_labels = [],transform =None, classToNum = {},superclassToNum = {}, sampleSize = 1.0,classList = {'0','1','2','3','4','5','6','7','8','9'}):
		"""
		root_dir : The testing/training data set which contains subfolders.
		sampleSize needs to be between 0 and 1
		"""

		#Raw Data
		self.root_dir = root_dir
		self.transform = transform
		self.classToNum = classToNum
		self.classToSuperClassMap = {}
		self.superClassSet = superclassToNum
		self.all_Superlabels = []
		self.all_labels = []
		self.all_image_paths = glob.glob(root_dir+"/**/*.png", recursive=True)
		self.all_Classlabels = [fadd.split('/')[-2] for fadd in self.all_image_paths ]
		self.all_SuperClasslabels = [fadd.split('/')[-3] for fadd in self.all_image_paths ]
		# self.classes = set(self.all_Classlabels)
		self.classList = classList
		# pdb.set_trace()
		# print('Class To Num = %s'%self.classToNum)
		# print('ID of self.classToNum == ID of classToNum : %s'%(id(self.classToNum)==id(classToNum)))
		for fadd in self.all_image_paths:
			if fadd.split('/')[-2] in self.classList:
				if fadd.split('/')[-2] not in self.classToNum:
					# print(fadd.split('/')[-3]+'\t'+fadd.split('/')[-2])
					self.classToNum.update({fadd.split('/')[-2]: len(self.classToNum)})
					self.all_labels.append(self.classToNum[fadd.split('/')[-2]])


					if fadd.split('/')[-3] not in self.superClassSet:
						self.superClassSet.update({fadd.split('/')[-3]:len(self.superClassSet)})

					### mapping the class to super class
					self.classToSuperClassMap.update({self.classToNum[fadd.split('/')[-2]]:self.superClassSet[fadd.split('/')[-3]] })
					self.all_Superlabels.append(self.superClassSet[fadd.split('/')[-3]])


				else:
					# print(fadd.split('/')[-3]+'\t'+fadd.split('/')[-2])
					# if fadd.split('/')[-3] == 'flowers':
					# 	print('flowers found!')
					self.all_labels.append(self.classToNum[fadd.split('/')[-2]])

					if fadd.split('/')[-3] not in self.superClassSet:
						self.superClassSet.update({fadd.split('/')[-3]:len(self.superClassSet)})

					self.classToSuperClassMap.update({self.classToNum[fadd.split('/')[-2]]:self.superClassSet[fadd.split('/')[-3]] })
					self.all_Superlabels.append(self.superClassSet[fadd.split('/')[-3]])
			else:
				print('Not in Class List : '+fadd.split('/')[-3]+'\t'+fadd.split('/')[-2])



		self.numToClass = {v: k for k, v in self.classToNum.items()}
		self.superClassSetReverse = {v: k for k, v in self.superClassSet.items()}
		self.classes = set(self.all_labels)		

		# preprocessing
		self.counter = Counter(self.all_labels)
		self.sampleSize = sampleSize


		self.LabelToIndex = self.generateLabelsToIndex()
		self.sampledLabelToIndex = self.generateSampledLabelsToIndex()


		# Sampled Data
		self.all_sampled_idx = self.generateSampledIdx() 
		self.all_sampled_labels = [self.all_labels[i] for i in self.all_sampled_idx] # to be used for all labels
		self.all_sampled_super_labels = [self.all_Superlabels[i] for i in self.all_sampled_idx] # to be used for all labels
		self.sampled_counter = Counter(self.all_sampled_labels)
		self.all_sampled_image_paths = [self.all_image_paths[i] for i in self.all_sampled_idx]


		# Class Distilled - Need to test extensively
		self.classIndicies = [index for index,element in enumerate(self.all_sampled_labels) if self.numToClass[element] in self.classList]#list of indices of relevant training examples
		self.classSampledLabels = [self.all_sampled_labels[i] for i in self.classIndicies] 
		self.superclassSampledLabels = [self.all_sampled_super_labels[i] for i in self.classIndicies] 
		self.classSampledImagePaths = [self.all_sampled_image_paths[i] for i in self.classIndicies]






	def __len__(self):
		""" returns the total number of files"""
		return len(self.classSampledImagePaths)


	def __getitem__(self,idx):
		""" return image for the given index"""
		imgAdd = self.classSampledImagePaths[idx]
		img = Image.open(imgAdd).convert('RGB')
		img = transforms.ToTensor()(img)
		# pdb.set_trace(0)
		return (img,int(self.classSampledLabels[idx]))


	# Helper Functions	

	def getLabels(self):
		return self.all_labels

	def generateLabelsToIndex(self):
		LabelToIndex = {}
		for idx,label in enumerate(self.all_labels):
			if label not in LabelToIndex:
				LabelToIndex.update({label:[idx]})
			else:
				LabelToIndex[label].append(idx)
		return LabelToIndex


	def getLabelsToIndex(self):
		return self.LabelToIndex

	def generateSampledLabelsToIndex(self):
		sampledLabelToIndex = {}
		for label, idx_list in self.LabelToIndex.items():
			indices = random.sample(range(len(idx_list)), int(len(idx_list)*self.sampleSize))
			sampledLabelToIndex.update({label: [idx_list[i] for i in sorted(indices)]})

		return sampledLabelToIndex


	def generateSampledIdx(self):
		all_sampled_idx = []
		for label,idx_list in self.sampledLabelToIndex.items():
			all_sampled_idx += idx_list
		return all_sampled_idx


	def getInputChannels(self):
		return 3

## Check Data loader Implementations

## From kuangliu github:
## https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
def get_mean_and_std(dataset):
	'''Compute the mean and std value of dataset.'''
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	print('==> Computing mean and std..')
	for inputs, targets in dataloader:
		for i in range(3):
			mean[i] += inputs[:,i,:,:].mean()
			std[i] += inputs[:,i,:,:].std()
	mean.div_(len(dataset))
	std.div_(len(dataset))
	return mean, std



# trans = transforms.Compose([transforms.ToTensor()])

### Subtract Mean for only sampled classes Later on

# trainMean,TrainVar = get_mean_and_std(Dataset(root_dir = rootDir))
# testMean, testVar = get_mean_and_std(Dataset(root_dir = rootDirTest))
transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(15),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# transforms.Normalize(tuple([float(x) for x in trainMean]),tuple([float(x) for x in TrainVar]))
	])


transform_test = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# transforms.Normalize(tuple([float(x) for x in testMean]),tuple([float(x) for x in testVar]))
	])

if classBased == 'False':
	CIFARtrain = Dataset(root_dir = rootDir,sampleSize = samplingFactor, transform = transform_train)
	lenTrain = len(CIFARtrain)
	print("Training Data: Loading sampling based loader\n")
	CIFARval1 = Dataset(root_dir = rootDirTest,sampleSize = samplingFactorTest, transform = transform_test)
	lenVal1 = len(CIFARval1)
	print("Probing Data: Loading sampling based loader\n")
	numClasses = len(CIFARtrain.classToNum)
	# pdb.set_trace()
elif classBased == 'True':
	CIFARtrain = DatasetClassBased(root_dir = rootDir, sampleSize = samplingFactor, transform = transform_train, classList = classListTrain)
	lenTrain = len(CIFARtrain)
	numClasses = len(classListTrain)
	print("Training Data: Loading Class based loader\n")
	# pdb.set_trace()
	if classListTrain == classListTest:
		print("Training and Testing Classes are the Same, Therefore using training dictionary for testing")
		classToNumDict = CIFARtrain.classToNum
		superclassToNumDict = CIFARtrain.superClassSet
		CIFARval1 = DatasetClassBased(root_dir = rootDirTest,sampleSize = samplingFactorTest, transform = transform_test, classToNum = classToNumDict, superclassToNum = superclassToNumDict ,classList = classListTest)
	else:
		print("Training and Testing Classes are not the Same, Therefore not using training dictionary for testing")
		CIFARval1 = DatasetClassBased(root_dir = rootDirTest,sampleSize = samplingFactorTest, transform = transform_test ,classToNum = {}, superclassToNum = {} ,classList = classListTest)#classToNum = {}, superclassToNum = {} ,
	lenVal1 = len(CIFARval1)
	# pdb.set_trace()
	print("Probing Data: Loading Class based loader\n")

	


# MNISTval2 = MNISTpngDataset(root_dir = rootDirVal2,sampleSize = samplingFactor, transform = trans)


# lenVal1 = len(CIFARval1)
# lenVal2 = len(MNISTval2)

# pdb.set_trace()




train_loader = torch.utils.data.DataLoader(dataset=CIFARtrain,
										   batch_size=train_batch_size,
										   shuffle=True)

val1_loader = torch.utils.data.DataLoader(dataset=CIFARval1,
										  batch_size=test_batch_size,
										  shuffle=False)


# val2_loader = torch.utils.data.DataLoader(dataset=MNISTval2,
# 										  batch_size=test_batch_size,
# 										  shuffle=False)


class LeNetCifar10(nn.Module):
	def __init__(self,numClasses = 10):
		super(LeNetCifar10, self).__init__()
		self.numClasses = numClasses
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, self.numClasses)

	def forward(self, x):
		out = FU.relu(self.conv1(x))
		out = x1 = FU.max_pool2d(out, 2)
		out = FU.relu(self.conv2(out))
		out = x2 = FU.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = x3 = FU.relu(self.fc1(out))
		out = x4 = FU.relu(self.fc2(out))
		out = self.fc3(out)
		return FU.log_softmax(out),(x1,x2,x3,x4)


	def countLayers(self):
		return 5
	def countPosLayers(self):
		return 4



class LeNet5MNIST(nn.Module):

	def __init__(self,numClasses = 10):
		super(LeNet5MNIST, self).__init__()
		self.numClasses = numClasses
		self.conv1 = nn.Conv2d(1, 6, kernel_size=5,padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.fc1 = nn.Linear(400, 120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,self.numClasses)

	def forward(self, x):
		in_size = x.size(0)
		x = x1 = self.pool1(FU.relu(self.conv1(x))) #This output needed
		x = x2 = self.pool2(FU.relu(self.conv2(x))) # This output needed
		x = x.view(in_size, -1)  # flatten the tensor
		x = x3 = FU.relu(self.fc1(x))
		x = x4 = FU.relu(self.fc2(x))
		x = x5 = self.fc3(x)
		return FU.log_softmax(x),(x1,x2,x3,x4)

	def countConvLayers(self):
		return 5
	def countPosLayers(self):
		return 4

# model = NetVGG11(evalMode = 'pool')

# VGG online 
if networkType == 'vgg':
	class NetVGG11(nn.Module): # github VGG, use transforms

		def __init__(self,num_classes=100): # get it checked by tushar.
			super(NetVGG11, self).__init__()
			self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
			self.bn1 = nn.BatchNorm2d(64)
			self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
			self.bn2 = nn.BatchNorm2d(64)
			#Maxpool2d comes here
			self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
			self.bn3 = nn.BatchNorm2d(128) # remove bn redundancies later
			self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
			self.bn4 = nn.BatchNorm2d(128)
			#Maxpool2d comes here
			self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
			self.bn5 = nn.BatchNorm2d(256)
			self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
			self.bn6 = nn.BatchNorm2d(256)
			self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
			self.bn7 = nn.BatchNorm2d(256)
			#Maxpool2d comes here
			self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
			self.bn8 = nn.BatchNorm2d(512)
			self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
			self.bn9 = nn.BatchNorm2d(512)
			self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
			self.bn10 = nn.BatchNorm2d(512)
			#Maxpool2d comes here
			self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
			self.bn11 = nn.BatchNorm2d(512)
			self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
			self.bn12 = nn.BatchNorm2d(512)
			self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
			self.bn13 = nn.BatchNorm2d(512)

			self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
			# self.dropout = nn.Dropout()
			self.fc = nn.Linear(512, num_classes)
			# self.evalMode = evalMode
			# nn.ReLU(inplace=True),
			# self.lin2 = nn.Linear(4096, 4096),
			# nn.ReLU(inplace=True),
			# self.lin3 = nn.Linear(4096, num_classes)
			# self.fc = nn.Linear(320, 10)

		def forward(self, x):
			in_size = x.size(0)
			x = x1 = FU.relu(self.bn1(self.conv1(x))) #This output needed
			# x = x2 = FU.relu(self.bn2(self.conv2(x))) # This output needed
			x = x3 = self.mp(x) # This output needed
			x = x4 = FU.relu(self.bn3(self.conv3(x))) # This output needed
			# x = x5 = FU.relu(self.bn4(self.conv4(x))) # This output needed
			x = x6 = self.mp(x) # This output needed
			x = x7 = FU.relu(self.bn5(self.conv5(x))) # This output needed
			x = x8 = FU.relu(self.bn6(self.conv6(x))) # This output needed
			# x = x9 = FU.relu(self.bn7(self.conv7(x)))
			x = x10 = self.mp(x) # This output is needed 
			x = x11 = FU.relu(self.bn8(self.conv8(x))) # This output is needed 
			x = x12 = FU.relu(self.bn9(self.conv9(x)))# This output is needed 
			# x = x13 = FU.relu(self.bn10(self.conv10(x)))# This output is needed 
			x = x14 = self.mp(x)# This output is needed 
			x = x15 = FU.relu(self.bn11(self.conv11(x)))# This output is needed 
			x = x16 = FU.relu(self.bn12(self.conv12(x)))# This output is needed 
			# x = x17 = FU.relu(self.bn13(self.conv13(x)))# This output is needed 
			x = x18 = self.mp(x)# This output is needed 



			x = x.view(in_size, -1)  # flatten the tensor
			# x = x.view(x.size(0), 256 * 6 * 6) # I have no idea

			# x = self.fc(x)
			# x = x6 = FU.relu(self.lin1(self.dropout()))# no clue whatesoever, though I need every ReLU
			# x = x7 = FU.relu(self.lin2(self.dropout()))
			x = x19 = self.fc(x)
			# if self.evalMode == 'all':
				# return FU.log_softmax(x),(x1,x3,x4,x6,x7,x8,x10,x11,x12,x14,x15,x16,x18,x19,FU.log_softmax(x))
			# elif self.evalMode == 'pool':
			return FU.log_softmax(x),(x3,x6,x10,x14,x18)

		def countConvLayers(self):
			# if self.evalMode == 'all':
			# 	return 15
			# elif self.evalMode == 'pool':
			return 1

		def countPosLayers(self):
			return 5
			


# model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)


# model = ResNet(BasicBlock, [2,2,2,2])




# Inception net from some pytorch implementation online
elif networkType == 'googlenet':

	class Inception(nn.Module):
		def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
			super(Inception, self).__init__()
			# 1x1 conv branch
			self.b1 = nn.Sequential(
				nn.Conv2d(in_planes, n1x1, kernel_size=1),
				nn.BatchNorm2d(n1x1),
				nn.ReLU(True),
			)

			# 1x1 conv -> 3x3 conv branch
			self.b2 = nn.Sequential(
				nn.Conv2d(in_planes, n3x3red, kernel_size=1),
				nn.BatchNorm2d(n3x3red),
				nn.ReLU(True),
				nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
				nn.BatchNorm2d(n3x3),
				nn.ReLU(True),
			)

			# 1x1 conv -> 5x5 conv branch
			self.b3 = nn.Sequential(
				nn.Conv2d(in_planes, n5x5red, kernel_size=1),
				nn.BatchNorm2d(n5x5red),
				nn.ReLU(True),
				nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
				nn.BatchNorm2d(n5x5),
				nn.ReLU(True),
				nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
				nn.BatchNorm2d(n5x5),
				nn.ReLU(True),
			)

			# 3x3 pool -> 1x1 conv branch
			self.b4 = nn.Sequential(
				nn.MaxPool2d(3, stride=1, padding=1),
				nn.Conv2d(in_planes, pool_planes, kernel_size=1),
				nn.BatchNorm2d(pool_planes),
				nn.ReLU(True),
			)

		def forward(self, x):
			y1 = self.b1(x)
			y2 = self.b2(x)
			y3 = self.b3(x)
			y4 = self.b4(x)
			return torch.cat([y1,y2,y3,y4], 1)


	class GoogLeNet(nn.Module):
		def __init__(self,num_classes=100):
			super(GoogLeNet, self).__init__()
			self.pre_layers = nn.Sequential(
				nn.Conv2d(3, 192, kernel_size=3, padding=1),
				nn.BatchNorm2d(192),
				nn.ReLU(True),
			)

			self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
			self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

			self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

			self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
			self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
			self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
			self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
			self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

			self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
			self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

			self.avgpool = nn.AvgPool2d(8, stride=1)
			self.linear = nn.Linear(1024, num_classes)

		def forward(self, x):
			out = x1 = self.pre_layers(x)
			out = x2 = self.a3(out)
			out = x3 = self.b3(out)
			out = x4 = self.maxpool(out)
			out = x5 = self.a4(out)
			out = x6 =self.b4(out)
			out = x7 = self.c4(out)
			out = x8 = self.d4(out)
			out = x9 = self.e4(out)
			out = x10 = self.maxpool(out)
			out = x11 = self.a5(out)
			out = x12 = self.b5(out)
			out = x13 = self.avgpool(out)
			out = out.view(out.size(0), -1)
			out = x14 = self.linear(out)
			out = x15 = FU.log_softmax(out)
			return out,(x2, x3 ,x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)

		def countConvLayers(self):
			return 1

		def countPosLayers(self):
			return 14


elif networkType == 'densenet':

	class Bottleneck(nn.Module):
		def __init__(self, nChannels, growthRate):
			super(Bottleneck, self).__init__()
			interChannels = 4*growthRate
			self.bn1 = nn.BatchNorm2d(nChannels)
			self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
								bias=False)
			self.bn2 = nn.BatchNorm2d(interChannels)
			self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
								padding=1, bias=False)

		def forward(self, x):
			out = self.conv1(FU.relu(self.bn1(x)))
			out = self.conv2(FU.relu(self.bn2(out)))
			out = torch.cat((x, out), 1)
			return out

	class SingleLayer(nn.Module):
		def __init__(self, nChannels, growthRate):
			super(SingleLayer, self).__init__()
			self.bn1 = nn.BatchNorm2d(nChannels)
			self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
								padding=1, bias=False)

		def forward(self, x):
			out = self.conv1(FU.relu(self.bn1(x)))
			out = torch.cat((x, out), 1)
			return out

	class Transition(nn.Module):
		def __init__(self, nChannels, nOutChannels):
			super(Transition, self).__init__()
			self.bn1 = nn.BatchNorm2d(nChannels)
			self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
								bias=False)

		def forward(self, x):
			out = self.conv1(FU.relu(self.bn1(x)))
			out = FU.avg_pool2d(out, 2)
			return out


	class DenseNet(nn.Module):
		def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
			super(DenseNet, self).__init__()

			nDenseBlocks = (depth-4) // 3
			if bottleneck:
				nDenseBlocks //= 2

			nChannels = 2*growthRate
			self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
								bias=False)
			self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
			nChannels += nDenseBlocks*growthRate
			nOutChannels = int(math.floor(nChannels*reduction))
			self.trans1 = Transition(nChannels, nOutChannels)

			nChannels = nOutChannels
			self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
			nChannels += nDenseBlocks*growthRate
			nOutChannels = int(math.floor(nChannels*reduction))
			self.trans2 = Transition(nChannels, nOutChannels)

			nChannels = nOutChannels
			self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
			nChannels += nDenseBlocks*growthRate

			self.bn1 = nn.BatchNorm2d(nChannels)
			self.fc = nn.Linear(nChannels, nClasses)

			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					m.weight.data.normal_(0, math.sqrt(2. / n))
				elif isinstance(m, nn.BatchNorm2d):
					m.weight.data.fill_(1)
					m.bias.data.zero_()
				elif isinstance(m, nn.Linear):
					m.bias.data.zero_()

		def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
			layers = []
			for i in range(int(nDenseBlocks)):
				if bottleneck:
					layers.append(Bottleneck(nChannels, growthRate))
				else:
					layers.append(SingleLayer(nChannels, growthRate))
				nChannels += growthRate
			return nn.Sequential(*layers)

		def forward(self, x):
			out = x1 = self.conv1(x)
			out = x2 = self.trans1(self.dense1(out))
			out = x3 = self.trans2(self.dense2(out))
			out = x4 = self.dense3(out)
			out = x5 = FU.avg_pool2d(FU.relu(self.bn1(out)), 8)
			out = x6 = out.view(x.size(0), -1) # check whether x.size(0) is batch size or not
			out = x7 = self.fc(out)
			out = x8 = FU.log_softmax(out)
			return out,(x2,x3,x5,x7,x8)

		def countConvLayers(self):
			return 5

		def countPosLayers(self):
			return 5


# elif networkType == 'resnet':
else:
	# Resnet
	# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
	"""resnet in pytorch
	[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
		Deep Residual Learning for Image Recognition
		https://arxiv.org/abs/1512.03385v1
	"""
	class BasicBlock(nn.Module):
		"""Basic Block for resnet 18 and resnet 34
		"""

		#BasicBlock and BottleNeck block 
		#have different output size
		#we use class attribute expansion
		#to distinct
		expansion = 1

		def __init__(self, in_channels, out_channels, stride=1):
			super().__init__()

			#residual function
			self.residual_function = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(out_channels * BasicBlock.expansion)
			)

			#shortcut
			self.shortcut = nn.Sequential()

			#the shortcut output dimension is not the same with residual function
			#use 1*1 convolution to match the dimension
			if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
				self.shortcut = nn.Sequential(
					nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(out_channels * BasicBlock.expansion)
				)
			
		def forward(self, x):
			return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

	class BottleNeck(nn.Module):
		"""Residual block for resnet over 50 layers
		"""
		expansion = 4
		def __init__(self, in_channels, out_channels, stride=1):
			super().__init__()
			self.residual_function = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
				nn.BatchNorm2d(out_channels * BottleNeck.expansion),
			)

			self.shortcut = nn.Sequential()

			if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
				self.shortcut = nn.Sequential(
					nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
					nn.BatchNorm2d(out_channels * BottleNeck.expansion)
				)
			
		def forward(self, x):
			return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
		
	class ResNet(nn.Module):

		def __init__(self, block, num_block, num_classes=100):
			super().__init__()

			self.in_channels = 64

			self.conv1 = nn.Sequential(
				nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(inplace=True))
			#we use a different inputsize than the original paper
			#so conv2_x's stride is 1
			self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
			self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
			self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
			self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
			self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
			self.fc = nn.Linear(512 * block.expansion, num_classes)

		def _make_layer(self, block, out_channels, num_blocks, stride):
			"""make resnet layers(by layer i didnt mean this 'layer' was the 
			same as a neuron netowork layer, ex. conv layer), one layer may 
			contain more than one residual block 
			Args:
				block: block type, basic block or bottle neck block
				out_channels: output depth channel number of this layer
				num_blocks: how many blocks per layer
				stride: the stride of the first block of this layer
			
			Return:
				return a resnet layer
			"""

			# we have num_block blocks per layer, the first block 
			# could be 1 or 2, other blocks would always be 1
			strides = [stride] + [1] * (num_blocks - 1)
			layers = []
			for stride in strides:
				layers.append(block(self.in_channels, out_channels, stride))
				self.in_channels = out_channels * block.expansion
			
			return nn.Sequential(*layers)

		def forward(self, x):
			output = x1 = self.conv1(x)
			output = x2 = self.conv2_x(output)
			output = x3 = self.conv3_x(output)
			output = x4 = self.conv4_x(output)
			output = x5 = self.conv5_x(output)
			output = x6 = self.avg_pool(output)
			output = output.view(output.size(0), -1)
			output = x7 = self.fc(output)

			return FU.log_softmax(output),(x4,x5,x6,x7,FU.log_softmax(output)) # This will be used for SDM paper.

		def countPosLayers(self):
			return 5



# model = LeNet5MNIST(numClasses)
if networkType == 'googlenet':
	print("Using GoogLeNet")
	model = GoogLeNet(numClasses)
	print("Number of Layers analyzed : %s"%model.countPosLayers())
elif networkType == 'resnet':
	print("Using ResNet18")
	model = ResNet(BasicBlock, [2, 2, 2, 2], numClasses)
	print("Number of Layers analyzed : %s"%model.countPosLayers())
elif networkType == 'vgg':
	print("Using VGG11")
	model = NetVGG11(numClasses)
	print("Number of Layers analyzed : %s"%model.countPosLayers())
elif networkType =='densenet':
	print("Using DenseNet")
	model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=numClasses)
	print("Number of Layers analyzed : %s"%model.countPosLayers())
else: 
	print("Invalid input. Using Resnet")
	model = ResNet(BasicBlock, [2, 2, 2, 2], numClasses)
	print("Number of Layers analyzed : %s"%model.countPosLayers())

model = model.cuda()
# model = AlexNet()

# This might change depending on the network
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.2)

def train(epoch):
	model.train()
	total_loss = 0
	for batch_idx, (data, target, superTarget, datapath) in enumerate(train_loader):
		# pdb.set_trace()
		# data, target = Variable(data), Variable(target)
		data, target = Variable(data).cuda(), Variable(target).cuda() # GPU
		optimizer.zero_grad()
		output,act = model(data)
		# pdb.set_trace()
		# output = model(data)
		loss = FU.nll_loss(output, target)
		total_loss += loss*len(data)
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

	print('Avg. Loss over the epoch: %f'%(total_loss/len(train_loader.dataset)))
	return total_loss/len(train_loader.dataset)




def testval1(): # add X-val later
	model.eval()
	val1_loss = 0
	correct = 0
	for data, target, superTarget, datapath in val1_loader:
		# data, target = Variable(data, volatile=True), Variable(target)
		data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # gpu
		output,act = model(data)
		# output = model(data)
		# sum up batch loss
		# pdb.set_trace()
		val1_loss += FU.nll_loss(output, target).item()
		# get the index of the max log-probability
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	val1_loss /= len(val1_loader.dataset)
	print('\nVal set 1: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val1_loss, correct, len(val1_loader.dataset),
		100. * correct / len(val1_loader.dataset)))
	return val1_loss,100. * correct / len(val1_loader.dataset)



# def probeVal1():
# 	# model.cpu()
# 	model.eval()
# 	D = [[] for j in range(CIFARval1.getInputChannels())]
# 	A = [[] for i in range(model.countPosLayers())]
# 	# A1 = []
# 	# A2 = []
# 	targetList = []
# 	pathList = []
# 	for data, target, datapath in val1_loader:
# 		targetList.append(int(target))
# 		pathList.append(datapath)
# 		data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
# 		# pdb.set_trace()
# 		inputImage = data.data.cpu().numpy()
# 		rawImage = inputImage[0]#first element of the unit sized batch - type numpy ndarray
# 		# print(rawImage.shape)
# 		#see the raw image
# 		# pdb.set_trace()
# 		for i in range(len(D)):
# 			# pdb.set_trace()
# 			D[i].append(rawImage[i]) 
# 		# D1.append(rawImage) 
# 		numRows = inputImage.shape[2]
# 		numCols = inputImage.shape[3]
# 		output,act = model(data)
# 		for j in range(len(act)):
# 			# pdb.set_trace()
# 			A[j].append(act[j].data.cpu().numpy()[0])
# 		# pdb.set_trace()
# 	# model.cuda()
# 	return D,A,targetList,pathList


def probeVal1():
	''' Issue Spotted 10 June 2020: Getting out a single matrix tends to falter
	with [0] at the end of act[j].data.cpu().numpy()[0] '''
	# model.cpu()
	model.eval()
	D = [[] for j in range(CIFARval1.getInputChannels())]
	A = [[] for i in range(model.countPosLayers())]
	# A1 = []
	# A2 = []
	targetList = []
	superClassList = []
	pathList = []
	for data, target, superTarget, datapath in val1_loader:
		targetList.append(int(target))
		superClassList.append(int(superTarget))
		pathList.append(datapath)
		data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
		# pdb.set_trace()
		inputImage = data.data.cpu().numpy()
		rawImage = inputImage[0]#first element of the unit sized batch - type numpy ndarray
		# print(rawImage.shape)
		#see the raw image
		# pdb.set_trace()
		for i in range(len(D)):
			# pdb.set_trace()
			D[i].append(rawImage[i]) 
		# D1.append(rawImage) 
		numRows = inputImage.shape[2]
		numCols = inputImage.shape[3]
		output,act = model(data)

		if len(act) > 1:
			for j in range(len(act)):
				# if j == len(act) - 1:
				# 	pdb.set_trace()
				A[j].append(act[j].data.cpu().numpy()[0])
				
		elif len(act) == 1:
			for j in range(len(act)):
				# if j == len(act) - 1:
				# 	pdb.set_trace()
				A[j].append(act[j].data.cpu().numpy())

		# pdb.set_trace()
	# model.cuda()
	return D,A,targetList,superClassList,pathList



def listOfArraysToMatByCol(loa):
	m = len(loa[0])
	n = len(loa)
	M = np.zeros((m,n))
	for idx, array in enumerate(loa):
		M[:,idx] = array
		# pdb.set_trace()
	return M



def genInputForTF(D,A):
	Dtemp = [[] for j in range(len(D))]
	Atemp = [[] for i in range(len(A))]
	# pdb.set_trace()
	for i in range(len(D)):
		Dtemp[i] = listOfArraysToMatByCol([tensor.flatten() for tensor in D[i]])
	for j in range(len(A)):
		Atemp[j] = listOfArraysToMatByCol([tensor.flatten() for tensor in A[j]])
	return Dtemp,Atemp






# Factorization

def chooseRank(listOfInputMatrices, thres = 0.8):
	M = sum(listOfInputMatrices)
	s = np.linalg.svd(M, compute_uv = False)
	total = sum(s)
	runner = 0
	for i, singVal in enumerate(s):
		runner += singVal
		if runner/total >= thres:
			return i+1


def norm(x):
	"""Dot product-based Euclidean norm implementation
	See: http://fseoane.net/blog/2011/computing-the-vector-norm/
	"""
	return math.sqrt(squared_norm(x))


# def analyzeF(F,labelList):
# 	vF = np.flipud(np.sort(F,axis = 0,kind = 'mergesort'))
# 	iF = np.flipud(np.argsort(F,axis = 0,kind = 'mergesort'))
# 	lF = np.zeros(iF.shape)
# 	for index,val in np.ndenumerate(iF):
# 		lF[index] = labelList[int(val)]

# 	return vF,iF,lF


# def generateReportF(vF,lF):
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
			imgC.__imul__(255.999/imgC.max())
			# pdb.set_trace()
			image.append(np.uint8(imgC))
		if l != 1:
			image = tuple(image)
			M = np.dstack(image)
		elif l == 1:
			M = image[l-1]
		# pdb.set_trace()
		floatMatrixToGS(M,os.path.join(path,'Latent Factor: '+str(imgNum)+'.jpg')) # removing magnification


def floatMatrixToHeatMapSNS(matrix,saveAs,dpi = 1200,cmap = 'hot', annot = False,linewidths = 0,fmt="f",annotFontSize = 10):#Pyplot
	ax = sns.heatmap(matrix,cmap = cmap, annot = annot,annot_kws={"size": annotFontSize},linewidths = linewidths,fmt=fmt)
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


def generateLatentActivations(O,path):
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
		suffix2 = os.path.join(path,'LayerLatentDimSimilarity%d.jpg'%layer )
		suffix3 = os.path.join(path,'LayerLatentCosineSimilarity%d.jpg'%layer )
		# try:
		# 	os.makedirs(suffixpt1)
		# except:
		# 	pass
		# for numImages in range(numLatentImages):
		# 	suffixpt1 = os.path.join(path,'LayerNum%d'%layer )
		# 	try:
		# 		os.makedirs(suffixpt1)
		# 	except:
		# 		pass
		# 	suffixpt2 = 'LatentDim:%d.jpg'%numImages
		# 	address = os.path.join(path,suffixpt1,suffixpt2)
		# 	vecL = O[layer][:,numImages]
		# 	mat = np.stack((vecL,vecL,vecL,vecL,vecL),axis = 1)
		# 	mat.__imul__(255.999/mat.max())
		# 	floatMatrixToGS(np.uint8(mat),address,25,4)
		# rowMaxes = np.amax(O[layer],axis = 0)
		# OlayerTilda = np.true_divide(O[layer],rowMaxes)
		# OlayerTilda.__imul__(255.999)
		# floatMatrixToGS(np.uint8(OlayerTilda),suffixpt1,2,10)
		floatMatrixToHeatMapSNS(O[layer],suffix1)
		floatMatrixToHeatMapSNS(O[layer].T@O[layer],suffix2)
		floatMatrixToHeatMapSNS(cosineSimilarity(O[layer].T), suffix3,annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)

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



def plotLists(dol,saveTo,legends = ['max','mean','min']):
	numLists = len(dol)
	plt.figure()
	for l in dol:
		plt.plot(dol[l])
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

def numToClassText(numToClassDict,saveTo):
	with open(saveTo,'w') as fH:
		for i in range(len(numToClassDict)):
			fH.write(numToClassDict[i]+'\n')


def clusterAnalysis(numClusters,clusterLabels,dataLabels,numToClassDict):
	d = {}
	# total = len(clusterLabels)
	for clusterID in range(numClusters):
		if clusterID not in d:
			d.update({clusterID:[]})

	for i,clusterNum in enumerate(clusterLabels):
		# print('ID : %d\tClass : %s'%(i,numToClassDict[dataLabels[i]]))
		d[clusterNum].append(numToClassDict[dataLabels[i]])


	for clusterID in d:
		d[clusterID] = Counter(d[clusterID])

	return d


def spectralDictToFile(clusterDict,saveTo,phrase = "Cluster"):
	
	with open(saveTo,'w') as fH:
		for clusterID in clusterDict:
			total = sum(clusterDict[clusterID].values())
			fH.write(phrase+" %s: "%clusterID)
			for testClass,score in clusterDict[clusterID].most_common():
				fH.write("\tGroup: %s - Fraction: %f"%(testClass,score/total))
			fH.write("\n")


def revClusterAnalysis(clusterLabels,dataLabels,numToClassDict):
	d = {}
	# total = len(clusterLabels)
	for classID in numToClassDict.values():
		if classID not in d:
			d.update({classID:[]})

	for i,classID in enumerate([numToClassDict[num] for num in dataLabels]):
		# print('ID : %d\tClass : %s'%(i,numToClassDict[dataLabels[i]]))
		d[classID].append(clusterLabels[i])


	for classID in d:
		d[classID] = Counter(d[classID])

	return d

def createImageFolders(clusterLabels,inputImagePaths,outputFolder):

	for i,clusterID in enumerate(clusterLabels):

		folderName = "Cluster-%s"%clusterID
		
		try:
			os.makedirs(os.path.join(outputFolder,folderName))
		except:
			pass
		
		fname = inputImagePaths[i].split('/')[-1]
		shutil.copy2(inputImagePaths[i],os.path.join(outputFolder,folderName,fname))


def classMask(l): 
		 m = np.zeros((len(l),len(l))) 
		 d = {i:v for i,v in enumerate(l)} 
		 for i in range(len(l)): 
			 for j in range(len(l)): 
				 if d[i] == d[j]: 
					 m[i,j] =1 
					  
		 return m 

def analyzeAffinity(Aff,labelList):
	vF = np.flipud(np.sort(Aff,axis = 0,kind = 'mergesort'))
	iF = np.flipud(np.argsort(Aff,axis = 0,kind = 'mergesort'))
	lF = np.zeros(iF.shape)
	for index,val in np.ndenumerate(iF):
		# pdb.set_trace()
		lF[index] = labelList[int(val)]

	return vF,iF,lF

def analyzeAffinityMatrixPerClass(Matrix,labelList,numToClassDict):
	labels = list(set(labelList))
	numClass = len(labels)
	ansDict = dict()

	# Setting up ans
	for label in labels:
		if label not in ansDict:
			ansDict.update({numToClassDict[label]: {numToClassDict[lab]:0 for lab in labels}})
		else:
			print("ERROR - Label Repeats: %s"%label)
	
	n,m = Matrix.shape

	for row in range(n):
		rowDataLabel = labelList[row]
		# rowSum = np.sum(Matrix[row,:])
		for col in range(m):
			colDataLabel = labelList[col]
			weight = Matrix[row,col]
			ansDict[numToClassDict[rowDataLabel]][numToClassDict[colDataLabel]] += weight
	
	return ansDict


def parseDictForPlot(d):
	xTicks = []
	yVals = []
	for xTick,yVal in d.items():
		xTicks.append(xTick)
		yVals.append(yVal)
	
	xVals = list(range(len(yVals)))

	return xTicks,yVals,xVals


def plotDict(xTicks,yVals,xVals,title,filePath):
	plt.figure()
	# plt.set_facecolor('white')
	plt.rcParams['axes.facecolor'] = 'white'

	plt.xticks(xVals, xTicks)
	# plt.plot(xVals, yVals)
	plt.bar(xVals, yVals)

	plt.xlabel('Class')
	plt.ylabel('Cumulative weight')
	plt.title(title)

	plt.savefig(filePath)
	plt.clf()


def ccaFile(ccaDict,filePath):
	for Class,distDict in ccaDict.items():
		xTicks,yVals,xVals = parseDictForPlot(distDict)
		plotDict(xTicks,yVals,xVals,'Conditional Distribution for Class - '+Class,os.path.join(filePath,'Class-%s.png'%Class))


def plotLists(lol,saveTo,legends = ['max','mean','min'], xlabel = 'Epochs', ylabel = 'Modularity', title = 'Modularity vs Epochs'):
	numLists = len(lol)
	plt.figure()
	plt.title(title, fontsize=20)
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(ylabel, fontsize=18)
	for l in lol:
		plt.plot(l)
	plt.legend(legends,loc = 'best')
	plt.savefig(saveTo)
	plt.clf()


netLossTraining = []
netLossVal1 = []
netAccVal1 = []
E = []
# mutualCoherenceSF = {}

# for i in range(model.countPosLayers()):
# 		mutualCoherenceSF[i] = [[] for j in range(3)]

		
# mutualCoherenceCF = [[] for j in range(3)]
# classmeanKLPerEpoch = []
# FKLmeanPerEpoch = []
# modularityListDict = {}
# for i in range(model.countPosLayers()):
# 		modularityListDict[i] = []

# modularityListDictNX = {}
# for i in range(model.countPosLayers()):
# 		modularityListDictNX[i] = []

# modularityListDictSC = {}
# for i in range(model.countPosLayers()):
# 		modularityListDictSC[i] = []


# eigenValListDict = {}
# for i in range(model.countPosLayers()):
# 		eigenValListDict[i] = []

# O matrix KNN analysis CF
algebraicConnectivityList = []
globalEfficiencyList = []
avgClusteringList = []
localEfficiencyList = []
O_matKNNCoupledAnalysis = {}
O_matAffinityMatSparsity = {}

# def countNeuralLayers(self):
# 		return 1

for i in range(model.countPosLayers()):
		O_matKNNCoupledAnalysis[i] = [[] for j in range(6)]

for i in range(model.countPosLayers()):
		O_matAffinityMatSparsity[i] = [[] for j in range(3)]


for epoch in range(0,numEpochs):
	epochFolder = 'Epoch Num %d'%epoch
	try:
		os.makedirs(os.path.join(outputFolderName,epochFolder))
	except:
		pass

	plt.figure()
	fig,axes = plt.subplots(1,1,figsize=(7,7))
	E.append(epoch) 
	trainLoss = train(epoch)
	scheduler.step()
	netLossTraining.append(float(trainLoss))
	# testval1()
	if CIFARtrain.classToNum == CIFARval1.classToNum: #classListTrain == classListTest or classBased=='False':
		testval1Loss, testval1Acc = testval1()
		netLossVal1.append(float(testval1Loss))
		netAccVal1.append(float(testval1Acc))
	# else if classBased=='False':
	# 	testval1Loss, testval1Acc = testval1()
	# 	netLossVal1.append(float(testval1Loss))
	# 	netAccVal1.append(float(testval1Acc))


	axes.set_ylabel("NLL Loss")
	axes.set_xlabel("Epochs")
	axes.set_title("Loss vs Epochs")
	# axes[0].set_xlim([-1,1])
	# axes[0].set_ylim([-0.5,4.5])
	axes.semilogy(E, netLossTraining, color = 'r', linewidth=1, label = "Training Loss")
	if CIFARtrain.classToNum == CIFARval1.classToNum: #classListTrain == classListTest or classBased=='False':
		axes.semilogy(E, netLossVal1, color = 'g', linewidth=1, label = "Validation Loss") ## For testing validation, set the dictionary of val set the same as test set
	axes.legend(loc = "upper right")
	axes.autoscale()

	plt.savefig(os.path.join(outputFolderName,epochFolder,'LossVsEpochs'))
	plt.clf()


	
	# #Accuracy Plots
	if CIFARtrain.classToNum == CIFARval1.classToNum:#classListTrain == classListTest or classBased=='False':
		plt.figure()
		fig,axes = plt.subplots(1,1,figsize=(7,7))
		axes.set_ylabel("Validation Accuracy")
		axes.set_xlabel("Epochs")
		axes.set_title("Accuracy vs Epochs")
		# axes[0].set_xlim([-1,1])
		# axes[0].set_ylim([-0.5,4.5])
		axes.plot(E, netAccVal1, color = 'r', linewidth=1, label = "Validation Accuracy")
		# axes[1].semilogy(E, netLossVal1, color = 'g', linewidth=1, label = "Validation Loss")
		axes.legend(loc = "upper right")
		axes.autoscale()
		plt.savefig(os.path.join(outputFolderName,epochFolder,'AccuracyVsEpochs'))
		plt.clf()


	torch.save(model,os.path.join(outputFolderName,epochFolder,NetworkName+"_Epoch-%d"%epoch))
	numToClassText(CIFARtrain.numToClass,os.path.join(outputFolderName,epochFolder,"numToClass.txt"))
