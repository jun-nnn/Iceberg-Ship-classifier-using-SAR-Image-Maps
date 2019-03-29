import scipy
import cv2
import numpy as np

def statsOfGradients(image):
   m, n = image.shape
   gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
   gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
   mag, ang = cv2.cartToPolar(gx, gy)
   mean_mag, mean_ang = np.sum(mag)/(m*n), np.sum(ang)/(m*n)

   std_mag, std_ang = np.std(mag), np.std(ang)
   skew_mag, skew_ang = scipy.stats.skew(mag), scipy.stats.skew(ang)
   kurt_mag, kurt_ang = scipy.stats.kurtosis(mag), scipy.stats.kurtosis(ang)
   return mean_mag, mean_ang, std_mag, std_ang, skew_mag, skew_ang, kurt_mag, kurt_ang
