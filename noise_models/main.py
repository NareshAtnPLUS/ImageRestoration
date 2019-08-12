import cv2 as cv
import numpy as np
from .noises import NoiseModels

def run():
	path = "data/"
	imgpath =  path + "nplus.jpg"
	img = cv.imread(imgpath, 1)
	cv.imshow('Original Image',img)
	cv.waitKey(0)
	salt_pepper = NoiseModels.impulse_noise(img,prob=0.1).astype(np.uint8)
	gauss = NoiseModels.gaussian_noise(img).astype(np.uint8)
	exp = NoiseModels.exponential_noise(img).astype(np.uint8)
	unifm = NoiseModels.uniform_noise(img).astype(np.uint8)

	cv.imshow('Salt and Pepper Noise',salt_pepper)
	cv.imshow('Gaussian',gauss)	
	cv.imshow('Exponential',exp)
	cv.imshow('Uniform',unifm)
	cv.waitKey(0)
