import numpy as np 
import cv2
import random
class NoiseModels:

    def impulse_noise(image,prob=0.1):
        '''
        Also Called Salt and Pepper Noise
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output.astype(np.uint8)

    def gaussian_noise(img):
        mean,std = 0.0,5.0
        noisy_img = img + 4*np.random.normal(mean,std,img.shape)
        return noisy_img.astype(np.uint8)
    def exponential_noise(img):
        noisy_img = img + np.exp(3.5)
        return noisy_img.astype(np.uint8)
    def uniform_noise(img):
        noisy_img = img + np.random.uniform(img.shape)
        return noisy_img.astype(np.uint8)