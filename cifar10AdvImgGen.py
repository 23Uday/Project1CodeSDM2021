from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as FU
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader, sampler
import glob
import PIL
from PIL import Image
from collections import Counter
import random
import pdb

epsilons = [0, 0.001, 0.01, .1]
train_batch_size = 64
test_batch_size = 1
use_cuda=True

parser = argparse.ArgumentParser() # argparser object
parser.add_argument("rootDir", help = "Enter the name of root folder which containts the data subfolders :",type = str)
parser.add_argument("rootDirTest", help = "Enter the name of root folder which containts the test data subfolders :",type = str)
parser.add_argument("outputFolderName", help = "Enter the name(Path) of the Output Folder :", type = str)
parser.add_argument("NetworkName", help = "Enter the name(Path) of the network file :", type = str)
parser.add_argument("samplingFactor", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("samplingFactorTest", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("lr", help = "learning rate for SGD to be used: ", type = float)
parser.add_argument("wd", help = "Enter the weight decay: ", type = float)
parser.add_argument("numEpochs", help = "Enter the number of epochs: ", type = int)
parser.add_argument('opt',help = "Enter the optimization algorithm", type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))


args = parser.parse_args()
rootDir = args.rootDir
rootDirTest = args.rootDirTest
outputFolderName = args.outputFolderName
if outputFolderName == 'pwd':
	outputFolderName = os.getcwd()
NetworkName = args.NetworkName
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
classListTestString = """[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck]"""
# classListTestString = """[beaver, dolphin, otter, seal, whale, 
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


outputFolderName = outputFolderName+"lr:%s-wd:-%s"%(lr,wd)

try:
	os.makedirs(outputFolderName)
except:
	pass

# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
	Deep Residual Learning for Image Recognition
	https://arxiv.org/abs/1512.03385v1
"""


class dataset(Dataset):
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

# trainMean,TrainVar = get_mean_and_std(dataset(root_dir = rootDir))
# testMean, testVar = get_mean_and_std(dataset(root_dir = rootDirTest))
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

CIFARtrain = dataset(root_dir = rootDir,sampleSize = samplingFactor, transform = transform_train)
lenTrain = len(CIFARtrain)
print("Training Data: Loading sampling based loader\n")
CIFARval1 = dataset(root_dir = rootDirTest,sampleSize = samplingFactorTest, transform = transform_test)
lenVal1 = len(CIFARval1)
print("Probing Data: Loading sampling based loader\n")
numClasses = len(CIFARtrain.classToNum)

if CIFARtrain.classToNum == CIFARval1.classToNum:
	print("Training and testing classes same")
else:
	print("Training and testing classes are NOT same")


train_loader = torch.utils.data.DataLoader(dataset=CIFARtrain,
										   batch_size=train_batch_size,
										   shuffle=True)

val1_loader = torch.utils.data.DataLoader(dataset=CIFARval1,
										  batch_size=test_batch_size,
										  shuffle=False)



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

		return FU.log_softmax(output)

	def countPosLayers(self):
		return 3



model = ResNet(BasicBlock, [2, 2, 2, 2], numClasses)
model = model.cuda()
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
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
		output = model(data)
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
		output = model(data)
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

for epoch in range(0,numEpochs):
	trainLoss = train(epoch)
	scheduler.step()
	if CIFARtrain.classToNum == CIFARval1.classToNum: #classListTrain == classListTest or classBased=='False':
		testval1Loss, testval1Acc = testval1()

def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon*sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	# Return the perturbed image
	return perturbed_image


# def test( model, device, test_loader, epsilon ):

# 	# Accuracy counter
# 	correct = 0
# 	initIncorrect = 0
# 	adv_examples = []

# 	# Loop over all examples in test set
# 	for data, target in test_loader:

# 		# Send the data and label to the device
# 		data, target = data.to(device), target.to(device)

# 		# Set requires_grad attribute of tensor. Important for Attack
# 		data.requires_grad = True

# 		# Forward pass the data through the model
# 		output = model(data)
# 		init_pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

# 		# If the initial prediction is wrong, dont bother attacking, just move on
# 		if init_pred.item() != target.item():
# 			initIncorrect += 1
# 			continue

# 		# Calculate the loss
# 		loss = FU.nll_loss(output, target)

# 		# Zero all existing gradients
# 		model.zero_grad()

# 		# Calculate gradients of model in backward pass
# 		loss.backward()

# 		# Collect datagrad
# 		data_grad = data.grad.data

# 		# Call FGSM Attack
# 		perturbed_data = fgsm_attack(data, epsilon, data_grad)

# 		# Re-classify the perturbed image
# 		output = model(perturbed_data)

# 		# Check for success
# 		final_pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
# 		if final_pred.item() == target.item():
# 			correct += 1
# 			# Special case for saving 0 epsilon examples
# 			if (epsilon == 0) and (len(adv_examples) < 5):
# 				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
# 				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
# 		else:
# 			# Save some adv examples for visualization later
# 			if len(adv_examples) < 5:
# 				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
# 				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

# 	# Calculate final accuracy for this epsilon
# 	final_acc = correct/float(len(test_loader))
# 	print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

# 	# Return the accuracy and an adversarial example
# 	return final_acc, adv_examples



def test( model, device, test_loader, epsilon ):

	# Accuracy counter
	correct = 0
	incorrect = 0
	adv_examples = []
	adv_examples_incorr = []
	#model.train()
	inaccurate = 0

	# Loop over all examples in test set
	for data, target in test_loader:

		# Send the data and label to the device
		data, target = data.to(device), target.to(device)

		# Set requires_grad attribute of tensor. Important for Attack
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)
		init_pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

		# If the initial prediction is wrong, dont bother attacking, just move on
		if init_pred.item() != target.item():
			inaccurate += 1
			continue

		# Calculate the loss
		loss = FU.nll_loss(output, target)

		# Zero all existing gradients
		model.zero_grad()

		# Calculate gradients of model in backward pass
		loss.backward()

		# Collect datagrad
		data_grad = data.grad.data

		# Call FGSM Attack
		perturbed_data = fgsm_attack(data, epsilon, data_grad)

		# Re-classify the perturbed image
		output = model(perturbed_data)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		if final_pred.item() == target.item():
			correct += 1
			# # Special case for saving 0 epsilon examples
			# if (epsilon == 0) and (len(adv_examples) < 5):
			# 	adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
			# 	adv_ex = np.transpose(adv_ex, (1,2,0))
			# 	adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
		elif final_pred.item() != target.item():
			# Save some adv examples for visualization later
			# if len(adv_examples) < 5:
			# pdb.set_trace()
			incorrect += 1
			adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
			adv_ex = np.transpose(adv_ex, (1,2,0))
			adv_examples_incorr.append( (init_pred.item(), final_pred.item(), adv_ex) )

	# pdb.set_trace()
	# Calculate final accuracy for this epsilon
	final_acc = correct/float(len(test_loader))
	print("Epsilon: {}\tTest Accuracy = {} / {} = {}\tError Rate = {} / {}\tInaccurate Predictions = {} / {}".format(epsilon, correct, len(test_loader), final_acc,incorrect, len(test_loader), inaccurate, len(test_loader)))

	# Return the accuracy and an adversarial example
	return final_acc, adv_examples, adv_examples_incorr


accuracies = []
examples = []
in_examples = []

# Run test for each epsilon
for eps in epsilons:
	acc, ex, in_ex = test(model, device, val1_loader, eps)
	accuracies.append(acc)
	examples.append(ex)
	in_examples.append(in_ex)
# pdb.set_trace()


def arrToImg(arr):
	Img = Image.fromarray(np.uint8(arr*255)).convert('RGB')
	return Img

def createAdvSet(lot, numToClass, saveTo ):
	d = dict()
	for t in lot:
		if numToClass[t[0]] not in d:
			d.update({numToClass[t[0]] : [arrToImg(t[2])]})
		else:
			d[numToClass[t[0]]].append(arrToImg(t[2]))
	
	for label in d:
		try:
			os.makedirs(os.path.join(saveTo,'AdversarialCifar10Set',"Adv-"+label))
		except:
			pass

		for i,img in enumerate(d[label]):
			img.save(os.path.join(saveTo,'AdversarialCifar10Set',"Adv-"+label,'%s-Image-%s.png'%(label,i)))

# createAdvSet(in_examples[-1], CIFARtrain.numToClass, outputFolderName)

for i,in_example in enumerate(in_examples):
	if i == 0:
		continue
	else:
		createAdvSet(in_example, CIFARtrain.numToClass, os.path.join(outputFolderName,'EPS:%s'%epsilons[i]))

# cnt = 0
# plt.figure(figsize=(8,10))
# for i in range(len(epsilons)):
# 	for j in range(len(in_examples[i])):
# 		cnt += 1
# 		plt.subplot(len(epsilons),len(in_examples[0]),cnt)
# 		plt.xticks([], [])
# 		plt.yticks([], [])
# 		if j == 0:
# 			plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
# 		orig,adv,ex = in_examples[i][j]
# 		plt.title("{} -> {}".format(orig, adv))
# 		# npex = ex.numpy()
# 		plt.imshow(ex, interpolation='nearest')
# plt.tight_layout()
# plt.savefig(os.path.join(outputFolderName,'testImage.png'))