import cv2
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from patchify import patchify
import matplotlib.pyplot as plt
from tifffile import imread, imsave
import sys

def exist_dir(file_path):
	'''
	Make sure the directory exists
	Parameters:
	file_path (str): path to the file

	Return:
	None
	'''
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

def Generate_patches(root_path, patch_size, path_s_masks = "./", path_s_imgs="./", mask_ext = "tif", img_ext = "tif", 
	do_return = False, verbose = True, save = True, do_show = False):
	'''
	Function to generate patches. The function has the purpose of cropping the images to get patches of size patch size. This function can
	either save the patches, return the patches or both.
	Parameters:

	root_path (str): path where to read images
	patch_size (str): size of the generated patch
	path_s_masks (str): path where to save masks
	path_s_imgs (str): path where to save imgs
	mask_ext (str): extension of the mask
	img_ext (str): extension of the img
	do_return (bool): return masks and patches
	verbose (bool): Verbose actions
	save (bool): Option to select if saving the imgs and patches is desired
	do_show (bool): Plot a few patches for visualization

	Return:

	(Optional -> do_return) img_dataset (np.array): img-patched dataset
	(Optional -> do_return) mask_dataset (np.array): masks-patched dataset
	'''
	#Make sure the paths exist
	if save:
		exist_dir(path_s_imgs)
		exist_dir(path_s_masks)
	if save or do_show:
		mask_show = []
		mask_dataset = []
		img_dataset = []
	for path, subdirs, files in os.walk(root_path):

		#Separate the path and grab the las name directory
		dirname = path.split(os.path.sep)[-1]
		#If the directory is images
		if dirname == "images":
			#Tile separation
			Tile = path.split(os.path.sep)[-2]
			Tile = Tile.replace(" ", "")
			#List all the images in that dir
			images = os.listdir(path)
			for i, image_name in enumerate(images):
				#If the files are images
				if image_name.endswith(img_ext):
					if mask_ext != "tif" or mask_ext != "tiff":
						#Read the RGB images
						img = cv2.imread(os.path.join(path,image_name), 1) #1 means RGB
					else:
						img = imread(os.path.join(path, image_name))

					Scale_factorX = (img.shape[1]//patch_size)*patch_size
					Scale_factorY = (img.shape[0]//patch_size)*patch_size

					#Convert image to np array
					image = Image.fromarray(img)

					#Crop image from 0,0 to the scale factor
					image = image.crop((0,0, Scale_factorX, Scale_factorY))
					image = np.array(image)

					if verbose:
						print("Start patching ", os.path.join(path,image_name))

					#Patch your image into smaller images of patch_size*patch_size*3, with a step of patch_size (no overlapping)
					#This will give you an array of images that contains N images that came from the original images
					patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

					#Extract the patches
					cont = 0
					for i in range(patches_img.shape[0]):
						for j in range(patches_img.shape[1]):
							#Delete the extra dim
							single_patch = patches_img[i,j,0,:,:]
							if save:
								if img_ext != "tif" or img_ext != "tiff":
									cv2.imwrite(os.path.join(path_s_imgs, Tile+"-"+str(cont)+image_name), single_patch)
								else:
									imsave(os.path.join(path_s_imgs, Tile+"-"+str(cont)+image_name), single_patch)
								cont += 1
							if do_return:
								img_dataset.append(single_patch)
							elif do_show:
								n = np.random.randint(0,10)
								if n == 3:
									img_dataset.append(single_patch)
									mask_show.append([Tile, image_name[:-4], i, j])


							
		elif dirname == "masks":
			#List all the images in that dir
			images = os.listdir(path)
			#Tile separation
			Tile = path.split(os.path.sep)[-2]
			Tile = Tile.replace(" ", "")
			for i, image_name in enumerate(images):
				#If the files are images
				if image_name.endswith(mask_ext):
					if mask_ext != "tif" or mask_ext != "tiff":
						#Read the RGB images
						img = cv2.imread(os.path.join(path,image_name), 1) #1 means RGB
						img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					else:
						img = imread(os.path.join(path, image_name))

					Scale_factorX = (img.shape[1]//patch_size)*patch_size
					Scale_factorY = (img.shape[0]//patch_size)*patch_size

					#Convert image to np array
					image = Image.fromarray(img)

					#Crop image from 0,0 to the scale factor
					image = image.crop((0,0, Scale_factorX, Scale_factorY))
					image = np.array(image)

					if verbose:
						print("Start patching ", os.path.join(path,image_name))

					#Patch your image into smaller images of patch_size*patch_size*3, with a step of patch_size (no overlapping)
					#This will give you an array of images that contains N images that came from the original images
					patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

					#Extract the patches
					cont = 0
					for i in range(patches_img.shape[0]):
						for j in range(patches_img.shape[1]):
							#Delete the extra dim
							single_patch = patches_img[i,j,0,:,:]
							if save:
								if mask_ext != "tif" or mask_ext != "tiff":
									cv2.imwrite(os.path.join(path_s_masks, Tile+"-"+str(cont)+image_name), single_patch)
								else:
									imsave(os.path.join(path_s_imgs, Tile+"-"+str(cont)+image_name), single_patch)
								cont += 1
							if do_return:
								mask_dataset.append(single_patch)
							elif do_show:
								if [Tile, image_name[:-4], i, j] in mask_show:
									mask_dataset.append(single_patch)
							
	
	
	#If do_show, show a few imgs and patches
	print(len(mask_dataset), len(img_dataset))
	if do_show:
		plt.figure(figsize=(12,12))
		c = 6
		for i in range(c):
			plt.subplot(2,c,i+1)
			plt.imshow(img_dataset[i])
			plt.axis("off")
		for j in range(c):
			plt.subplot(2,c,j+c+1)
			plt.imshow(mask_dataset[j])
			plt.axis("off")
		plt.show()

	#If do_return, return the dataset in an array
	if do_return:
		img_dataset = np.array(img_dataset)
		mask_dataset = np.array(mask_dataset)
		return img_dataset, mask_dataset

def main(root_path, path_s_imgs, path_s_masks):
	#root_path = r"D:\Datasets\aeri_dubai\Semantic segmentation dataset"
	#path_s_masks = r"D:\Datasets\aeri_dubai\patches\masks\."
	#path_s_imgs = r"D:\Datasets\aeri_dubai\patches\images\."
	patch_size = 256
	imgs, masks = Generate_patches(root_path, patch_size, path_s_masks, path_s_imgs, mask_ext = "png", img_ext = "jpg", 
			do_return = True, verbose = True, save = True, do_show=False)
	print(imgs.shape)

if __name__ == "__main__":
	sys.argv[1:]
	root_path = r'{}'.format(sys.argv[1])
	path_s_imgs = r'{}'.format(sys.argv[2])
	path_s_masks = r'{}'.format(sys.argv[3])
	print(root_path)
	main(root_path, path_s_imgs, path_s_masks)