# Project1CodeSDM2021
## Paper Data and Paper Results
[Link to Data Files](https://drive.google.com/drive/folders/1ALW3_nt317-QnAuZKXi0_EeGobFTkm-r?usp=sharing)

The above link contains 3 Items:-
1. CIFAR100Dataset : Contains zip file for CIFAR100 datasets
2. ExperimentResultsResNet18: Contains the results of an analysis of a ResNet-18 on CIFAR-100
3. A Pretrained ResNet-18 on CIFAR-100

## Requirements
1. Python 3.7.6 and above
2. Numpy 1.18.1 and above
3. Pytorch 1.4.0 and above
4. Scipy 1.4.1 and above

## Command Template to Run the experiment
Prototype Command:
```
python3 dissectSDM21KNNResNet18.py FullPathToCifar100SuperClass/train/ FullPathToCifar100SuperClass/valKDD/ PathToPretrainedResnet18/Resnet18Cifar100 OutputFolderDirectory ResNet18Cifar100 20 20 50 1 1 1 1 1 0.001 0.00001 100 sgd NNDSVD False False False
```
The parameter values shown were used for experimentation.

Use the following command to get help:
```
python3 dissectSDM21KNNResNet18.py -h
```
usage: dissectSDM21KNNResNet18.py [-h]                                  rootDir rootDirTest networkFile
                                  outputFolderName NetworkName Rank1 numGroups
                                  maxIters lmbdaF lmbdaTV lmbdaOrtho
                                  samplingFactor samplingFactorTest lr wd
                                  numEpochs {sgd,adam,rmsprop}
                                  {random,ID,NNDSVD} {True,False} {True,False}
                                  {True,False}

positional arguments:
-  rootDir             Enter the name of root folder which containts the data
                      subfolders :
 - rootDirTest         Enter the name of root folder which containts the test
                      data subfolders :
 - networkFile         Enter the name of root folder which containts the
                      Network :
-  outputFolderName    Enter the name(Path) of the Output Folder :
-  NetworkName         Enter the name(Path) of the network file :
-  Rank1               Enter Rank 1 :
-  numGroups           Enter the number of groups:
-  maxIters            Enter Max no. of iterations:
-  lmbdaF              Enter lambda F:
-  lmbdaTV             Enter lambda TV:
-  lmbdaOrtho          Enter orthogonality penalty:
-  samplingFactor      Enter the ratio of dataset to be used:
-  samplingFactorTest  Enter the ratio of dataset to be used:
-  lr                  learning rate for SGD to be used:
-  wd                  Enter the weight decay:
-  numEpochs           Enter the number of epochs:
-  {sgd,adam,rmsprop}  Enter the optimization algorithm
-  {random,ID,NNDSVD}  Enter the initialization for P
-  {True,False}        Enter if class based training is required
-  {True,False}        Enter true if Data matrices have to be ignored
-  {True,False}        Enter true if F matrix is to have group sparsity
