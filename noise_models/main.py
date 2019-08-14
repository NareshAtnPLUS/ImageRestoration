import cv2 as cv
import numpy as np
from .noises import NoiseModels
from .filters import *
filters = {'medianFilter':medianFilter,'min_max_filter':min_max_filter,'mean_filter':mean_filter,'harmonic_mean_filter':harmonic_mean_filter,'geometric_mean_filter':geometric_mean_filter}
noises = {'impulse_noise':NoiseModels.impulse_noise,'gaussian_noise':NoiseModels.gaussian_noise,'exponential_noise':NoiseModels.exponential_noise,'uniform_noise':NoiseModels.uniform_noise}
noise_imgs = {}
filterd_imgs = {}
class ImageRestoration:

	def __init__(self,image,kernal_size):
		self.img = image
		self.k_size = kernal_size

	def add_noises(self):
		for noise,noise_func in zip(noises,noises.values()):
			noise_imgs.update({noise:noise_func(self.img)})
	def apply_filters(self):
		for noise,noise_img in zip(noise_imgs,noise_imgs.values()):
			for filt,filt_func in zip(filters,filters.values()):
				print(f'Applying {filt} for {noise}')
				filterd_imgs.update({(noise+' '+filt):filt_func(noise_img,self.k_size)})
			print('\n\n')
		#print(filterd_imgs)
	def transforms(self):
		self.img1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		h, w = self.img1.shape[:2]
		vis0 = np.zeros((h,w), np.float32)
		vis0[:h, :w] = self.img1
		self.sine_tf = cv.dft(vis0)		
		self.isine_tf = cv.idft(self.sine_tf)
		self.vis1 = cv2.dct(vis0)
		self.ivis1 = cv2.idct(self.vis1)


	def processed_images(self):
		cv.imshow('Original Image',self.img)
		for noise,noise_img, in zip(noise_imgs,noise_imgs.values()):
			cv.imshow(noise,noise_img)
		for filterd,filterd_img in zip(filterd_imgs,filterd_imgs.values()):
			cv.imshow(filterd,filterd_img)
		cv.imshow('DCT',self.vis1)
		cv.imshow('IDCT',self.ivis1)
		cv.waitKey(0)		



def run():
	path = "data/"
	imgpath =  path + "image.jpg"
	img = cv.imread(imgpath, 1)

	img_res = ImageRestoration(img,kernal_size = 5)

	img_res.add_noises()

	img_res.apply_filters()
	img_res.transforms()
	img_res.processed_images()

	