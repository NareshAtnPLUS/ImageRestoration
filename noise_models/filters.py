import numpy as np 
import cv2 
def medianFilter(img,kernal_size):
	return cv2.medianBlur(img,kernal_size)


def boxFilter(img,kernal_size):
	return cv2.boxFilter(img,-1,(kernal_size,kernal_size))


def bilateralFilter(img,kernal_size):
	return cv2.bilateralFilter(img,kernal_size,75,75)