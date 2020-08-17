import PIL
from PIL import Image
from PIL import Image, ImageDraw, ImageFont, ImageColor
import shutil


def floatMatrixToGS(matrix,saveAs,magFactorY = 1,magFactorX = 1): # To Visualize P matrix
	""" SaveAs contains the filepath and filename"""
	# max = matrix.max()
	# matrix.__imul__(255/max)
	img = PIL.Image.fromarray(matrix)
	img = img.convert('RGB')
	img = img.resize((magFactorY*matrix.shape[0],magFactorX*matrix.shape[1]))
	img.save(saveAs)

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

			latentImage = []
			for c in range(numChannels):
				imgC = P[c][:,imageIndex].reshape(32,32)
				imgC.__imul__(255.999/imgC.max())
				imgC[imgC > 64] = 1
				imgC[imgC <= 64] = 0
				latentImage.append(np.uint8(imgC))

			if numChannels != 1:
				latentImage = tuple(latentImage)
				M = np.dstack(latentImage)
			elif numChannels == 1:
				M = latentImage[l-1]
			# fname = imagePath.split('/')[-3].strip()+'_'+imagePath.split('/')[-2].strip()+'_'+imagePath.split('/')[-1].strip().split('.')[0]
			# fname = str(imageNum+1)+'_'+fname+'_'+'Weight : %s'%str(vF[imageNum,LFNUM])+'_'+'Per : %s'%str(100*vF[imageNum,LFNUM]/sumFactor)+'_'+'Top_Weight : %s'+str(weightCollected)
			# shutil.copy2(imagePath,os.path.join(saveTo,'LatentFactor-%d'%LFNUM,fname+'.png'))