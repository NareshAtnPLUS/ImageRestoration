import cv2 as cv
import numpy as np
from .noises import NoiseModels
from .filters import *

class ImageRestoration:

	def __init__(self,image,kernal_size):
		self.img = image
		self.k_size = kernal_size

	def add_noises(self):
		self.salt_pepper = NoiseModels.impulse_noise(self.img,prob=0.1).astype(np.uint8)
		self.gauss = NoiseModels.gaussian_noise(self.img).astype(np.uint8)
		self.exp = NoiseModels.exponential_noise(self.img).astype(np.uint8)
		self.unifm = NoiseModels.uniform_noise(self.img).astype(np.uint8)
	
	def apply_filters(self):
		self.med_filter = medianFilter(self.salt_pepper,self.k_size)
		self.box_filter = boxFilter(self.salt_pepper,self.k_size)
		self.bilateral_filter = bilateralFilter(self.salt_pepper,self.k_size)


	def transforms(self):
		self.img1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		h, w = self.img1.shape[:2]
		vis0 = np.zeros((h,w), np.float32)
		vis0[:h, :w] = self.img1
		self.sine_tf = cv.dft(vis0)		
		self.isine_tf = cv.idft(self.sine_tf)
		self.vis1 = cv2.dct(vis0)
		self.ivis1 = cv2.idct(self.vis1)
		#print(vis1)
		# self.dct_image = cv.CreateMat(vis1.shape[0], vis1.shape[1], cv.CV_32FC3)
		# cv.CvtColor(cv.fromarray(vis1), self.dct_image, cv.CV_GRAY2BGR)


	def processed_images(self):
		cv.imshow('Original Image',self.img)
		cv.imshow('Salt and Pepper Noise',self.salt_pepper)
		cv.imshow('Filtered Salt and Pepper Noise with median Filter',self.med_filter)
		cv.imshow('Filtered Salt and Pepper Noise with box Filter(Mean Filter)',self.box_filter)
		cv.imshow('Exponential',self.exp)
		cv.imshow('Uniform',self.unifm)
		cv.imshow('Discrete Cosine Transform',self.vis1.astype(np.uint8))
		cv.imshow('Inverse Discrete Cosine Transform',self.ivis1.astype(np.uint8))
		cv.waitKey(0)		



def run():
	path = "data/"
	imgpath =  path + "nplus.jpg"
	img = cv.imread(imgpath, 1)

	img_res = ImageRestoration(img,kernal_size = 5)

	img_res.add_noises()
	img_res.apply_filters()
	img_res.transforms()
	img_res.processed_images()


	