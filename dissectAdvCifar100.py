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

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser() # argparser object
parser.add_argument("rootDir", help = "Enter the name of root folder which containts the data subfolders :",type = str)
parser.add_argument("rootDirTest", help = "Enter the name of root folder which containts the test data subfolders :",type = str)
# parser.add_argument("networkFile", help = "Enter the name of root folder which containts the Network :",type = str)
parser.add_argument("outputFolderName", help = "Enter the name(Path) of the Output Folder :", type = str)
parser.add_argument("NetworkName", help = "Enter the name(Path) of the network file :", type = str)
parser.add_argument("Rank1", help = "Enter Rank 1 :", type = int)
# parser.add_argument("Rank2", help = "Enter Rank 2 :", type = int)
# parser.add_argument("Rank3", help = "Enter Rank 2 :", type = int)
parser.add_argument("numGroups", help = "Enter the number of groups: ", type = int)
parser.add_argument("maxIters", help = "Enter Max no. of iterations: ", type = int)
# parser.add_argument("lmbdaSR", help = "Enter lambda S or R: ", type = float)
# parser.add_argument("lmbdaSimplex", help = "Enter lambda Simplex :", type = float)
parser.add_argument("lmbdaF", help = "Enter lambda F: ", type = float)
parser.add_argument("lmbdaTV", help = "Enter lambda TV: ", type = float)
parser.add_argument("lmbdaOrtho", help = "Enter orthogonality penalty: ", type = float)
parser.add_argument("samplingFactor", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("samplingFactorTest", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("lr", help = "learning rate for SGD to be used: ", type = float)
parser.add_argument("wd", help = "Enter the weight decay: ", type = float)
parser.add_argument("numEpochs", help = "Enter the number of epochs: ", type = int)
parser.add_argument('opt',help = "Enter the optimization algorithm", type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument('P_init',help = "Enter the initialization for P", type=str, default='random', choices=('random', 'ID', 'NNDSVD'))
parser.add_argument("classBased", help = "Enter if class based training is required ", type = str, default = 'False',choices = ('True','False'))
# parser.add_argument("classListTrain", help = "Enter the list of classes, eg [1,2,3,4]: ",type = str,default = "['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
# parser.add_argument("classListTest", help = "Enter the list of classes, eg [1,2,3,4]: ",type = str,default = "['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
parser.add_argument("actOnlyMode", help = "Enter true if Data matrices have to be ignored", type = str, default = 'False',choices = ('True','False'))
parser.add_argument("groupSparseF", help = "Enter true if F matrix is to have group sparsity", type = str, default = 'False',choices = ('True','False'))


args = parser.parse_args()


rootDir = args.rootDir
rootDirTest = args.rootDirTest

outputFolderName = args.outputFolderName
if outputFolderName == 'pwd':
	outputFolderName = os.getcwd()
NetworkName = args.NetworkName

Rank1 = args.Rank1
# Rank2 = args.Rank2
# Rank3 = args.Rank3
numGroups = args.numGroups
maxIters = args.maxIters
# lmbdaSR = args.lmbdaSR
# lmbdaSimplex = args.lmbdaSimplex
lmbdaF = args.lmbdaF
lmbdaTV = args.lmbdaTV
lmbdaOrtho = args.lmbdaOrtho
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
P_init=args.P_init
classBased = args.classBased
# classListTrainString = """[0,1,2,3,4,5,6,7,8,9]"""
classListTrainString = """[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck]"""
# classListTrainString = """[beaver, dolphin, otter, seal, whale, 
# aquarium_fish, flatfish, ray, shark, trout, 
# orchid, poppy, rose, sunflower, tulip, 
# bottle, bowl, can, cup, plate, 
# apple, mushroom, orange, pear, sweet_pepper, 
# clock, keyboard, lamp, telephone, television, 
# bed, chair, couch, table, wardrobe, 
# bee, beetle, butterfly, caterpillar, cockroach, 
# bear, leopard, lion, tiger, wolf, 
# bridge, castle, house, road, skyscraper, 
# cloud, forest, mountain, plain, sea, 
# camel, cattle, chimpanzee, elephant, kangaroo, 
# fox, porcupine, possum, raccoon, skunk, 
# crab, lobster, snail, spider, worm, 
# baby, boy, girl, man, woman, 
# crocodile, dinosaur, lizard, snake, turtle, 
# hamster, mouse, rabbit, shrew, squirrel, 
# maple_tree, oak_tree, palm_tree, pine_tree, willow_tree, 
# bicycle, bus, motorcycle, pickup_truck, train, 
# lawn_mower, rocket, streetcar, tank, tractor]"""
#args.classListTrain

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
groupSparseF = args.groupSparseF
if groupSparseF == 'True':
	groupSparseF = True
elif groupSparseF == 'False':
	groupSparseF = False


classListTrain = set([str(x.strip()) for x in classListTrainString.strip('[]').strip().split(',')]) # Make set for efficiency
classListTest = set([str(x.strip()) for x in classListTestString.strip('[]').strip().split(',')]) # Make set for efficiency

samplingAdd = 'samplingFactor'+str(samplingFactor)

# outputFolderName =  outputFolderName+'samplingFactor'+str(samplingFactor)
outputFolderName = outputFolderName+"Rank:%s-Numgroups:%s-Iters:%s-lmbdaF:%s-lmbdaTV:%s-ortho:%s-lr:%s-wd:%s-P_init-%s"%(Rank1,numGroups,maxIters,lmbdaF,lmbdaTV,lmbdaOrtho,lr,wd,P_init)
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
		self.all_sampled_image_paths = [self.all_image_paths[i] for i in self.all_sampled_idx]






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
		return (img,self.all_sampled_labels[idx])


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
		output = self.conv1(x)
		output = self.conv2_x(output)
		output = self.conv3_x(output)
		output = x1 = self.conv4_x(output)
		output = x2 = self.conv5_x(output)
		output = x3 = self.avg_pool(output)
		output = output.view(output.size(0), -1)
		output = self.fc(output)

		return FU.log_softmax(output),(x1,x2,x3) 

	def countPosLayers(self):
		return 3



# model = LeNet5MNIST(numClasses)


model = ResNet(BasicBlock, [2, 2, 2, 2], numClasses)
model = model.cuda()
# model = AlexNet()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.2)

def train(epoch):
	model.train()
	total_loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
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
	for data, target in val1_loader:
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



def probeVal1():
	# model.cpu()
	model.eval()
	D = [[] for j in range(CIFARval1.getInputChannels())]
	A = [[] for i in range(model.countPosLayers())]
	# A1 = []
	# A2 = []
	for data, target in val1_loader:
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
		for j in range(len(act)):
			# pdb.set_trace()
			A[j].append(act[j].data.cpu().numpy()[0])
		# pdb.set_trace()
	# model.cuda()
	return D,A



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
	img = img.convert('L')
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


def generateLatentActivations(O,path , title = "", xLabel= "", yLabel = "",):
	numLayers = len(O)
	numLatentImages,numDim = O[0].shape
	fontSize = 10
	annotations = True
	if numDim>100:
		annotations = False
	elif numDim>10:
		fontSize *= 10/numDim
	for layer in range(numLayers):
		# suffix1 = os.path.join(path,'LayerNum%d.jpg'%layer )
		# suffix2 = os.path.join(path,'LayerLatentDimSimilarity%d.jpg'%layer )
		suffix3 = os.path.join(path,'LayerLatentCosineSimilarity%d.jpg'%layer )
		# floatMatrixToHeatMapSNS(O[layer],suffix1)
		# floatMatrixToHeatMapSNS(O[layer].T@O[layer],suffix2)
		floatMatrixToHeatMapSNS(cosineSimilarity(O[layer].T), suffix3, title , xLabel, yLabel, annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)

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
				latentFactorImg = latentFactorImg.convert('L')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				latentFactorImgArr[latentFactorImgArr <= 10] = 0
				imgc0[latentFactorImgArr == 0] = 0
				imgc1[latentFactorImgArr == 0] = 0
				imgc2[latentFactorImgArr == 0] = 0

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



netLossTraining = []
netLossVal1 = []
netAccVal1 = []
E = []
mutualCoherenceSF = {}

for i in range(model.countPosLayers()):
		mutualCoherenceSF[i] = [[] for j in range(3)]

		
mutualCoherenceCF = [[] for j in range(3)]
classmeanKLPerEpoch = []
FKLmeanPerEpoch = []
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

	# this was added to speed up experiments
	if epoch == numEpochs-1:
		

		tupleOfData = probeVal1()
		D1,A1 = tupleOfData
		# pdb.set_trace()
		D,A = genInputForTF(D1,A1)
		# pdb.set_trace()
		if actOnlyMode == 'True':
			# D[0] = A[0] # Shift the first activation to D and keep the rest in A
			# A = A[1:]
			# # this will work with MNIST ONLY
			# # This will lead to a situation where D[0] of the original setup is P[0] here
			D = [] # emptying the D list
			# If D list is empty, P list should also be empty.


		# pdb.set_trace()
		# for i,mat in enumerate(D):
			# print("D[%d] rank : %f\t D[%d] shape : %s\n"%(i,np.linalg.matrix_rank(mat),i,mat.shape))

		# for i,mat in enumerate(A):
			# print("A[%d] rank : %f\t A[%d] shape : %s\n"%(i,np.linalg.matrix_rank(mat),i,mat.shape))

		# pdb.set_trace()


		os.chdir(outputFolderName) # getting to output folder
		


		

		##### Layer by Layer analysis

		analysisType = 'SingleFactorization'
		# setting up dictionary for single factorization mutual coherence
		
		

		for i,mat in enumerate(A):
			print("***** Analysis of A[%d] *****\n"%i)
			F,P,O,LOSSTOTAL,LOSSACT = cnmf([], [mat], Rank1, P_init = P_init, groupSparseF = groupSparseF, numGroups = numGroups, lmbdaF=lmbdaF,lmbdaTV = lmbdaTV, lmbdaOrtho=lmbdaOrtho, maxIter = maxIters, compute_fit = True)
			print("F shape : %s \t"%(F.shape,))

			try:
				os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-O[%d]'%i))
				os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F'))
			except:
				pass

			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-O[%d]'%i))
			generateLatentActivations(O,os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-O[%d]'%i),'Cosine Similarity w.r.t Neurons','Latent Factor','Latent Factor')

			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F'))
			generateLatentActivations([F.T],os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F'),'Cosine Similarity','Latent Factor','Latent Factor')
		# 	# pdb.set_trace()
		# 	MutualCoherence = mutualCoherence(F)
		# 	for j,array in enumerate(mutualCoherenceSF[i]):
		# 		array.append(MutualCoherence[j])

		# 	# pdb.set_trace()
		# 	# print("plotting",mutualCoherenceSF[i])
		# 	plotLists(mutualCoherenceSF[i],os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','Mutual-Coherence-matrix-F'))

		# 	# for j,layers in enumerate(mutualCoherenceSF):
		# 	# 	pdb.set_trace()
		# 	# 	plotLists(mutualCoherenceSF[layers],os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','Mutual-Coherence-matrix-F'))
		# 	# This needs to be changed when using class sampled
			if classBased == 'True':
				vF,iF,lF = analyzeF(F.T,CIFARval1.classSampledLabels)
				SvF,SiF,SlF = analyzeF(F.T,CIFARval1.superclassSampledLabels)
				topImagesPerLatentFactor(vF,iF,CIFARval1.classSampledImagePaths,os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','topImagesPerLatentFactor'))
			else:
				vF,iF,lF = analyzeF(F.T,CIFARval1.all_sampled_labels) # This needs to be fixed because we need index to class mapping,
				SvF,SiF,SlF = analyzeF(F.T,CIFARval1.all_sampled_super_labels)
				topImagesPerLatentFactor(vF,iF,CIFARval1.all_sampled_image_paths,os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','topImagesPerLatentFactor'))
			#F.T used because of the notational change
			# somewhere downstream to properly populate lF
				# NvF = normalize(vF,axis = 0)




		# 	# pdb.set_trace()
		# 	### Class Probabilities
		# 	classProbMat = vFlFToClassMat(vF,lF)
		# 	# superClassProbMat = vFlFToClassMat(SvF,SlF)

		# 	numDim = classProbMat.shape[0]
		# 	fontSize = 10
		# 	annotations = True
		# 	if numDim>100:
		# 		annotations = False
		# 	elif numDim>10:
		# 		fontSize *= 10/numDim

		# 	## Computing hellinger
		# 	classHellinger = pairwiseHellinger(classProbMat)
		# 	floatMatrixToHeatMapSNS(classHellinger, os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','LF-ClassHellinger'),annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)
			
		# 	# superClassHellinger = pairwiseHellinger(superClassProbMat)
		# 	# floatMatrixToHeatMapSNS(superClassHellinger, os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','LF-superClassHellinger'),annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)


		# 	## Computing bhattacharyya
		# 	classBhattacharyya = pairwiseBhattacharyya(classProbMat)
		# 	floatMatrixToHeatMapSNS(classBhattacharyya, os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','LF-ClassBhatt'),annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)
		# 	# superClassBhattacharyya = pairwiseBhattacharyya(superClassProbMat)
		# 	# floatMatrixToHeatMapSNS(superClassBhattacharyya, os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','LF-superClassBhatt'),annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)



		# 	## Computing KL
		# 	# FKL = pairwiseKL(F)
		# 	# FKLmeanPerEpoch.append(FKL.mean())


		# 	classKL = pairwiseKL(classProbMat)
		# 	# classmeanKLPerEpoch.append(classKL.mean())
		# 	floatMatrixToHeatMapSNS(classKL, os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','LF-ClassKL'),annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)
		# 	# superClassKL = pairwiseKL(superClassProbMat)
		# 	# floatMatrixToHeatMapSNS(superClassKL, os.path.join(outputFolderName,epochFolder,analysisType,'matrix-A[%d]'%i,'matrix-F','LF-superClassKL'),annot = annotations, fmt = "1.2f",cmap = 'viridis',annotFontSize = fontSize)

		# 	# plotLists([FKLmeanPerEpoch,classmeanKLPerEpoch], os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor,'meanClassKL.png'),legends = ['Unsupervised KL','Class Based KL'])
			
		# 	# pdb.set_trace()
			lod = generateReportF(vF,lF)
			slod = generateReportF(SvF,SlF)
			# pdb.set_trace()

			FReport(lod,os.getcwd(),CIFARval1.numToClass) # Writing F Report
			FReport(slod,os.getcwd(),CIFARval1.superClassSetReverse,'FReportSuperClass.txt') # Writing F Report
			FAdvReport(lod,os.getcwd(),CIFARval1.numToClass,'ClassBasedAdvReport.txt')
			FAdvReport(slod,os.getcwd(),CIFARval1.superClassSetReverse,'SuperClassBasedAdvReport.txt')

		



		os.chdir(os.path.join(outputFolderName,epochFolder))

		##### coupled factorization #####
		F,P,O,LOSSTOTAL,LOSSACT = cnmf(D, A, Rank1, P_init = P_init, groupSparseF = groupSparseF, numGroups = numGroups, lmbdaF=lmbdaF, lmbdaTV = lmbdaTV, lmbdaOrtho=lmbdaOrtho, maxIter = maxIters, compute_fit = True)
		print("F shape : %s \t"%(F.shape,))
		# pdb.set_trace()
		# ********** Generating Reports **********

		

		


		analysisType = 'CoupledFactorization'
		try:
			os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType))
		except:
			pass

		parentDirLatentImaging = 'LatentAnalysis'

		try:
			os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging))
		except:
			pass
			
		os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging))

		#Latent Images to be stored here
		if actOnlyMode == 'False':
			latentImages = 'latentImages'
			try:
				os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentImages))
			except:
				pass

			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentImages))
			generateLatentImages(P,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentImages))
			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging))


		# Neural activations latent map

			latentActivations = 'latentActivations'
			try:
				os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentActivations))
			except:
				pass

			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentActivations))
			generateLatentActivations(O,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentActivations),'Cosine Similarity w.r.t Neurons','Latent Factor','Latent Factor')
			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging))



		elif actOnlyMode == "True":

			latentActivations = 'latentActivations'
			try:
				os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentActivations))
			except:
				pass

			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentActivations))
			generateLatentActivations(P+O,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,latentActivations))
			# Because D[0] is A[0] of the original setup
			os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging))


		# Top Classes per latent Dimension

		topClassesLatentFactor = 'TopClassesPerLatentFactor'

		try:
			os.makedirs(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor))

		except:
			pass

		os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor))
		generateLatentActivations([F.T],os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor))
		
		MutualCoherence = mutualCoherence(F)
		for index,val in enumerate(MutualCoherence):
			mutualCoherenceCF[index].append(val)


		plotLists(mutualCoherenceCF,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor,'Mutual-Coherence-matrix-F'))
		# pdb.set_trace()

		# This needs to be changed when using class sampled
		if classBased == 'True':
			vF,iF,lF = analyzeF(F.T,CIFARval1.classSampledLabels)
			SvF,SiF,SlF = analyzeF(F.T,CIFARval1.superclassSampledLabels)
			topImagesPerLatentFactor(vF,iF,CIFARval1.classSampledImagePaths,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor,'topImagesPerLatentFactor','Raw'))
			topMaskedImagesPerLatentFactor(vF,iF,P,CIFARval1.classSampledImagePaths,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor,'topImagesPerLatentFactor','Masked'))
		else:
			vF,iF,lF = analyzeF(F.T,CIFARval1.all_sampled_labels) # This needs to be fixed because we need index to class mapping,
			SvF,SiF,SlF = analyzeF(F.T,CIFARval1.all_sampled_super_labels)
			topImagesPerLatentFactor(vF,iF,CIFARval1.all_sampled_image_paths,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor,'topImagesPerLatentFactor','Raw'))
			topMaskedImagesPerLatentFactor(vF,iF,P,CIFARval1.all_sampled_image_paths,os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging,topClassesLatentFactor,'topImagesPerLatentFactor','Masked'))
		#F.T used because of the notational change
		# somewhere downstream to properly populate lF
			# NvF = normalize(vF,axis = 0)


		### Class Probabilities
		classProbMat = vFlFToClassMat(vF,lF)
		superClassProbMat = vFlFToClassMat(SvF,SlF)

		numDim = classProbMat.shape[0]
		fontSize = 10
		annotations = True
		if numDim>100:
			annotations = False
		elif numDim>10:
			fontSize *= 10/numDim

		lod = generateReportF(vF,lF)
		slod = generateReportF(SvF,SlF)
		# pdb.set_trace()

		FReport(lod,os.getcwd(),CIFARval1.numToClass) # Writing F Report
		FReport(slod,os.getcwd(),CIFARval1.superClassSetReverse,'FReportSuperClass.txt') # Writing F Report
		FAdvReport(lod,os.getcwd(),CIFARval1.numToClass,'ClassBasedAdvReport.txt')
		FAdvReport(slod,os.getcwd(),CIFARval1.superClassSetReverse,'SuperClassBasedAdvReport.txt')

		os.chdir(os.path.join(outputFolderName,epochFolder,analysisType,parentDirLatentImaging))
