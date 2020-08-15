import argparse
import glob
import numpy as np
import os



parser = argparse.ArgumentParser() # argparser object
parser.add_argument("rootDir", help = "Enter the name of root folder which containts the data subfolders :",type = str)
parser.add_argument("rootDirTest", help = "Enter the name of root folder which containts the test data subfolders :",type = str)
parser.add_argument("outputFolderName", help = "Enter the name(Path) of the Output Folder :", type = str)
parser.add_argument("Rank1List", help = "Enter Rank 1 :", type = str,default = "[5,10,15,20,25,30,35,40,45,50,75,100]")
parser.add_argument("numGroupsList", help = "Enter the number of groups: ", type = str,default = "[2,3,5,10,15,20,30,40,50,75,100]")
parser.add_argument("maxIters", help = "Enter Max no. of iterations: ", type = int)
parser.add_argument("lmbdaFList", help = "Enter lambda F: ", type = str,default = "[0.01,0.1,1,10,100,1000]" )
parser.add_argument("samplingFactor", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("samplingFactorTest", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("lr", help = "learning rate for SGD to be used: ", type = float)
parser.add_argument("wd", help = "Enter the weight decay: ", type = float)
parser.add_argument("numEpochs", help = "Enter the number of epochs: ", type = int)
parser.add_argument('opt',help = "Enter the optimization algorithm", type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument('P_init',help = "Enter the initialization for P", type=str, default='random', choices=('random', 'ID', 'NNDSVD'))
parser.add_argument("classBased", help = "Enter if class based training is required ", type = str, default = 'False',choices = ('True','False'))
parser.add_argument("classList", help = "Enter the list of classes, eg [1,2,3,4]: ",type = str,default = "[0,1,2,3,4,5,6,7,8,9]")
parser.add_argument("actOnlyMode", help = "Enter true if Data matrices have to be ignored", type = str, default = 'False',choices = ('True','False'))


args = parser.parse_args()


def str2NumList(s,ftype):
	return [ftype(x) for x in s.strip('[]').strip().split(',')]


rootDir = args.rootDir
rootDirTest = args.rootDirTest

outputFolderName = args.outputFolderName
if outputFolderName == 'pwd':
	outputFolderName = os.getcwd()


Rank1List = str2NumList(args.Rank1List,int)
numGroupsList = str2NumList(args.numGroupsList,int)
maxIters = args.maxIters
lmbdaFList = str2NumList(args.lmbdaFList,float)
samplingFactor = args.samplingFactor
samplingFactorTest = args.samplingFactorTest
if samplingFactor > 1:
	samplingFactor = 1.0
elif samplingFactor < 0:
	samplingFactor = 1.0

lr = args.lr

wd = args.wd
numEpochs = args.numEpochs
opt = args.opt
P_init=args.P_init
classBased = args.classBased
classList = args.classList
actOnlyMode = args.actOnlyMode


classList = set([int(x) for x in classList.strip('[]').strip().split(',')]) # Make set for efficiency

samplingAdd = 'samplingFactor'+str(samplingFactor)
try:
	os.makedirs(outputFolderName)
except:
	pass


for rank in Rank1List:
	for numGroups in numGroupsList:
		for lmbdaF in lmbdaFList:
			command  = 'ipython3 dissect.py %s %s %s %d %d %d %f %f %f %f %f %d %s %s %s "[0,1,2,3,4,5,6,7,8,9]" %s'%(rootDir,rootDirTest,outputFolderName,rank,numGroups,maxIters,lmbdaF,samplingFactor,samplingFactorTest,lr,wd,numEpochs,opt,P_init,classBased,actOnlyMode)
			print("Running :\n%s"%command)
			os.system(command)