import os
import PIL
from PIL import Image
import numpy as np



def floatMatrixToGS(matrix,saveAs,magFactor = 1): # To Visualize P matrix
	""" SaveAs contains the filepath and filename"""
	max = matrix.max()
	matrix.__imul__(255/max)
	img = PIL.Image.fromarray(matrix)
	img = img.convert('RGB')
	img = img.resize((magFactor*matrix.shape[0],magFactor*matrix.shape[1]))
	img.save(saveAs)


mat1 = np.ones((100,50))
mat2 = np.zeros((100,50))
mat3 = np.ones((100,50))

M = np.dstack((mat1,mat2,mat3))

# floatMatrixToGS(M,os.path.join(os.getcwd(),'testImageRBChannels.jpg'),5)
