import os
import numpy as np
import pandas as pd
from skimage.util.montage import montage2d
import keras.preprocessing.image as prep
import matplotlib.pyplot as plt
import imutils
import cv2
import math
import random
import json
import augment

MAXMAG = 0
MAXANG = 0

def showimage(image1, image2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.matshow(image1)
    ax1.set_title(title1)
    ax2.matshow(image2)
    ax2.set_title(title2)

def hog(image):
    binnum = 16
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    minmag, maxmag = np.amin(mag),np.amax(mag)
    minang, maxang = np.amin(ang),np.amax(ang)
    #showimage(image,mag,"image","bins")
    #print(minmag,maxmag,minang,maxang)
    bins = np.int64(binnum*mag//(maxmag-minmag))
##    for i in range(len(bins)):
##        for j in range(len(bins[0])):
##            if bins[i,j] < 0:
##                print(i,j,bins[i,j])
    #m = np.bincount(bins.ravel())
    #print(bins.shape, len(m))
    angs = np.int64(binnum*ang//(maxang-minang))
    #bins, angs = np.int64(np.fabs(bins)), np.int64(np.fabs(angs))
    #a = np.bincount(angs.ravel())
    showimage(image,mag,"image","magnitude")
    showimage(image,ang,"image",'angle')
    #plt.show()
    bin_cells = bins[:37,:37], bins[37:,:37], bins[:37,37:], bins[37:,37:]
    ang_cells = angs[:37,:37], angs[37:,:37], angs[:37,37:], angs[37:,37:]
    hists = [np.bincount(b.ravel(), m.ravel(), 17) for b, m in zip(bin_cells, ang_cells)]
    #hists = [np.bincount(b.ravel(), m.ravel(), binnum) for b, m in zip(bins, angs)]
    
    #print(hists, len(hists))
    #for e in hists:
    #    print(len(e))
    hist = np.hstack(hists) #size 65
    #print(hist, len(hist))
    return hist
    #return m, a

def hog2(image):
    #image = data.astronaut()
    fd, hog_image = hog(image, orientations=8)#, pixels_per_cell=(16, 16))#,
                        #cells_per_block=(1, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

def findmax(imset):
    currmag, currang = 0, 0
    for image in imset:
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        if np.amax(ang) > currang:
            currang = np.amax(ang)
        if np.amax(mag) > currmag:
            currmag = np.amax(mag)
    return currmag, currang

def modify(image):
    L = []
    samplenum = 3
    for i in range(samplenum):
        L.append(np.amax(image[i,:]))
        L.append(np.amax(image[len(image)-1-i,:]))
    for i in range(samplenum):
        L.append(np.amax(image[:,i]))
        L.append(np.amax(image[:,len(image)-1-i]))
    cutoff = min(L)
        
    copy = image
    for i in range(len(copy)):
        for j in range(len(copy[0])):
            if copy[i,j] < cutoff:
                copy[i,j] = cutoff
    return copy


train_df, train_images = augment.load_and_format('train.json')
train_y_df, train_y = augment.load_and_format2('train.json')
print('training', train_df.shape, 'loaded', train_images.shape)


### training data overview
##fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
##obj_list = dict(ships = train_df.query('is_iceberg==0').sample(16).index,
##     icebergs = train_df.query('is_iceberg==1').sample(16).index)
##for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
##    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
##    ax1.set_title('%s Band 1' % obj_type)
##    ax1.axis('off')
##    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
##    ax2.set_title('%s Band 2' % obj_type)
##    ax2.axis('off')

#showimage(train_images[7,:,:,1],train_images[7,:,:,1],"","")


#descriptor_vector(train_images[100,:,:,0])
##for i in range(len(train_images)):
##    train_images[i,:,:,0] = modify(train_images[i,:,:,0])
##    train_images[i,:,:,1] = modify(train_images[i,:,:,1])
### modified data overview
##fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
##obj_list = dict(ships = train_df.query('is_iceberg==0').sample(16).index,
##     icebergs = train_df.query('is_iceberg==1').sample(16).index)
##for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
##    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
##    ax1.set_title('%s Band 1' % obj_type)
##    ax1.axis('off')
##    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
##    ax2.set_title('%s Band 2' % obj_type)
##    ax2.axis('off')

MAXMAG, MAXANG = findmax(train_images)
X = np.ndarray((len(train_images), 17*4*2))



#for i in range(101,105):
#    print(train_y[i])
for i in range(100, 101):   #(len(train_images)): 
    v1 = hog(train_images[i,:,:,0])
    v2 = hog(train_images[i,:,:,1])
##    if len(np.concatenate([v1,v2])) == 130:
##        print("130", i, len(v1), len(v2))
##    if len(np.concatenate([v1,v2])) == 131:
##        print("131", i, len(v1), len(v2))
    X[i] = np.concatenate([v1,v2])
print(X.shape)
plt.show()

