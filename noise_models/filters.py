import numpy as np 
import cv2 
from statistics import harmonic_mean
def medianFilter(img,kernal_size):
	return cv2.medianBlur(img,kernal_size)

def min_max_filter(img,kernal_size):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernal_size, kernal_size))
	min_img = cv2.erode(img,kernel)
	return cv2.dilate(min_img,kernel)
def mean_filter(img,kernal_size):
	return cv2.boxFilter(img,-1,(kernal_size,kernal_size))

def harmonic_mean_filter(img,kernal_size):
	harm = np.zeros_like(img)
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			harm[row,col] = (kernal_size**2)/((np.sum(1/(img[row:row+kernal_size, col:col+kernal_size] + 0.001) )))
	return np.uint8(harm)

def geometric_mean_filter(img,kernal_size):
	geom = np.zeros_like(img)
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			geom[row,col] = np.prod(img[row:row+kernal_size,col:col+kernal_size]**(1/kernal_size**2))
	return np.uint8(geom)
	
def bilateralFilter(img,kernal_size):
	return cv2.bilateralFilter(img,kernal_size,75,75)