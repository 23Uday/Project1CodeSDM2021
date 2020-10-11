# Project1CodeSDM2021
## Paper Data and Paper Results
[Link to Data Files](https://drive.google.com/drive/folders/1ALW3_nt317-QnAuZKXi0_EeGobFTkm-r?usp=sharing)

The above link contains 2 Folders:-
1. CIFAR100Dataset : Contains zip file for CIFAR100 datasets
2. ExperimentResultsResNet18: Contains the results of an analysis of a ResNet-18 on CIFAR-100

## Requirements
1. Python 3.7.6 and above
2. Numpy 1.18.1 and above
3. Pytorch 1.4.0 and above
4. Scipy 1.4.1 and above

## Command to Run the experiment
python3 dissectSDM21KNNResNet18.py /data/usain001/data/Cifar100SuperClass/train/ /data/usain001/data/Cifar100SuperClass/valKDD/ /data/usain001/outputs/Paper1/AdvTesting/Test3FullValAdvCifar100/lr\:0.001-wd\:-1e-05/Resnet18Cifar100 /data/usain001/outputs/Paper1/KNNanalysis/26Sep/Trial5SavedResNetFFF/ ResNet18Cifar100 20 20 50 1 1 1 1 1 0.001 0.00001 100 sgd NNDSVD False False False
