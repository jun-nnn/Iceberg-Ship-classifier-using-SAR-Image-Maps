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
from skimage.feature import hog
from skimage import data, exposure
from sklearn import decomposition
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import SVC
import scipy

MAXMAG = 0
MAXANG = 0

# load image data
def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images

# load y values
def load_and_format2(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['is_iceberg']], -1).reshape((1))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images

# rotate only the inscribed circle of the image
def rotate(image, angel):
    copy = np.copy(image)
    circled = copy
    edge = []
    r = len(image)/2
    for i in range(len(image)):
        edge.append([])
        for j in range(len(image[0])):
            if math.sqrt((i-r)**2+(j-r)**2) >= r-3:
                edge[i].append(circled[i][j])
            else:
                edge[i].append(0)
            if math.sqrt((i-r)**2+(j-r)**2) > r:
                circled[i][j] = 0
    rotated = imutils.rotate(circled, angel)
    for i in range(len(image)):
        for j in range(len(image[0])):
            if math.sqrt((i-r)**2+(j-r)**2) >= r-3:
                rotated[i][j] = edge[i][j]
    return rotated

# flip the image horizontally
def mirror(image):
    copy = np.copy(image)
    m = len(copy)
    n = len(copy[0])
    for i in range(m):
        for j in range(n//2):
            copy[i][j], copy[i][n-j-1] = copy[i][n-j-1], copy[i][j]
    return copy

# a backup method that is not used for now
def shift(image, width_rg=15, height_rg=15, u=0.5, v=1.0):

    if v < u:
        image = prep.random_shift(image, wrg=width_rg, hrg=height_rg, 
                                  row_axis=0, col_axis=1, channel_axis=2)

    return image

# augment images
def augment(imset, aug_size):
    # first augment by rotation
    n = len(imset)
    copy = np.copy(imset)
    augmented = np.zeros((n*(aug_size-1),75,75,2))
    angel = 360//aug_size
    m = 0
    print("Rotating images")
    for i in range(n):
        e = imset[i,:,:,:]
        for j in range(aug_size-1):
            m += 1
            if m%2000 == 0:
                print('{percent:.2%}'.format(percent=m/((aug_size-1)*n)))
            augmented[j*n+i,:,:,0] = rotate(e[:,:,0], angel*(j+1))
            augmented[j*n+i,:,:,1] = rotate(e[:,:,1], angel*(j+1))
    augmented = np.concatenate((copy, augmented))
    
    # then augment by mirroring
    mirrored = np.zeros((n*aug_size,75,75,2))
    n2 = n*aug_size
    m = 0
    print("\nFlipping images")
    for i in range(n2):
        e = augmented[i,:,:,:]
        m += 1
        if m%2000 == 0:
            print('{percent:.2%}'.format(percent=m/n2))
        mirrored[i,:,:,0] = mirror(e[:,:,0])
        mirrored[i,:,:,1] = mirror(e[:,:,1])
    augmented = np.concatenate((augmented, mirrored))
    print(augmented.shape)
    return augmented

# augment the binary y values
def augment_y(yset, n):
    augmented = np.zeros(len(yset)*n)
    for i in range(len(augmented)):
        augmented[i] = yset[i%len(yset)]
    print("augmented y", augmented.shape)
    return augmented

# pick a random image and test the augmented effects
def testaugmented(augmented, size):
    n = random.randint(0,size-1)
    m = random.randint(0,99)%2
    print("sample number: "+str(n), " band: "+str(m))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.matshow(augmented[n,:,:,m])
    ax1.set_title('original')
    ax2.matshow(augmented[n+size*2,:,:,m])
    ax2.set_title('60 degrees')

    fig, (ax3, ax4) = plt.subplots(1,2, figsize = (12, 6))
    ax3.matshow(augmented[n+size*12,:,:,m])
    ax3.set_title('original degress flipped')
    ax4.matshow(augmented[n+size*14,:,:,m])
    ax4.set_title('60 degress flipped')

    fig, (ax5, ax6) = plt.subplots(1,2, figsize = (12, 6))
    ax5.matshow(augmented[n+size*8,:,:,m])
    ax5.set_title('240 degrees')
    ax6.matshow(augmented[n+size*10,:,:,m])
    ax6.set_title('300 degrees')

    fig, (ax7, ax8) = plt.subplots(1,2, figsize = (12, 6))
    ax7.matshow(augmented[n+size*20,:,:,m])
    ax7.set_title('240 degrees flipped')
    ax8.matshow(augmented[n+size*22,:,:,m])
    ax8.set_title('300 degrees flipped')

    plt.show()

def load_and_augment():
    train_df, train_images = load_and_format('train.json')
    print('training', train_df.shape, 'loaded', train_images.shape)
    train_df.sample(3)
    train_y_df, train_y = load_and_format2('train.json')
    print('training', train_y_df.shape, 'loaded', train_y.shape)    
    # the augmented images of size (38496, 75, 75, 2)
    # second parameter is default as 12 for rotating 360/12=30 degrees
    augmented = augment(train_images, 12)
    # the augmented y values of size (38496, 1)
    augmented_y = augment_y(train_y, 24)
    testaugmented(augmented, len(train_images))
    return augmented, augmented_y

def showimage(image1, image2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.matshow(image1)
    ax1.set_title(title1)
    ax2.matshow(image2)
    ax2.set_title(title2)

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

def svm(X,Y):
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
    num=X.shape
    num=num[0]
    labels=Y
    #labels=train.is_iceberg.values
    kf=KFold(num,n_folds=10,shuffle=False)
    scores=cross_val_score(clf,X,labels,cv=10,scoring='accuracy')
    print(scores*100)
    scores=scores*100
    print("Mean of all scores is %d"%(scores.mean()))

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

def getHogDescriptor(image,binNumber = 16):
##   image = modify(image)
   gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
   gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
   mag, ang = cv2.cartToPolar(gx, gy)
##   print(mag)
   bins = np.int32(binNumber*ang/(2*np.pi))    # quantizing binvalues in (0...16)


# 69   
   bin_cells = [bins[20:50,20:50]\
               ,bins[:20, :50], bins[:50, 50:], \
               bins[50:, :50], bins[20:, :20]]
   mag_cells = [mag[20:50,20:50],\
               mag[:20, :50], mag[:50, 50:], \
               mag[50:, :50], mag[20:, :20]]

##   bin_cells = bins[10:60,10:60]\
##               ,bins[:10, :60], bins[:60, 60:], \
##               bins[60:, :60], bins[10:, :10]
##   mag_cells = mag[10:60,10:60],\
##               mag[:10, :60], mag[:60, 60:], \
##               mag[60:, :60], mag[10:, :10]

##   bin_cells = [bins[20:50,20:50]]
##   bincell2 = [bins[:20, :50], bins[:50, 50:], \
##               bins[50:, :50], bins[20:, :20]]
##   bin_edge = []
##   for e in bincell2:
##      bin_edge.append[i for row in e for i in row]
##   bin_cells.append(bin_edge)
##   
##   mag_cells = [mag[20:50,20:50]]\
##   magcell2 = [mag[:20, :50], mag[:50, 50:], \
##               mag[50:, :50], mag[20:, :20]]
##   mag_edge = []
##   for e in magcell2:
##      mag_edge.append[i for row in e for i in row]
##   mag_cells.append(mag_edge)

   
##   bin_cells = bins[30:40,30:40], bins[25:45,25:45], \
##               bins[20:50,20:50], bins[15:55,15:55]\
##
##   mag_cells = mag[30:40,30:40], mag[25:45,25:45], \
##               mag[20:50,20:50], mag[15:55,15:55],\

   #print((b.ravel(), m.ravel()) for b, m in zip(bin_cells, mag_cells))
##   hists = [np.hstack(b) for b in bin_cells]
##   hists = [np.hstack(a) for a in hists]
##   hists = [np.bincount(h, weights=None, minlength=16) for h in hists]
##
##
##   hists2 = [np.hstack(b) for b in mag_cells]
##   hists2 = [np.hstack(a) for a in hists2]
##   print(hists2)
##   hists2 = [np.bincount(h, weights=None, minlength=16) for h in hists2]
   #hists = [np.bincount(np.hstack(b), weights=None, minlength=16) for b in bin_cells]
   #print(hists)
##   hists = []
##   for b,m in zip(bin_cells, mag_cells):
##      thiscount = np.bincount(b.ravel(), m.ravel(), binNumber)
##      if len(thiscount) > 16:
##         thiscount[15] = sum(thiscount[15:])                          
##      hists.append(thiscount)
##
##   for i in range(16):
##      hists[1][i] += hists[2][i] + hists[3][i] + hists[4][i]
##   hists = hists[:2]
 
   hists = [np.bincount(b.ravel(), m.ravel(), binNumber) for b, m in zip(bin_cells, mag_cells)]
   hist = np.hstack(hists)     # hist is a 64 bit vector
   
   hist = np.array(hist,dtype=np.float32)
   print(len(hists[0]))
   plt.bar([i+1 for i in range(len(hists[0]))], hists[0], align='center', alpha=0.5)
   plt.xticks([i+1 for i in range(len(hists[0]))], [i+1 for i in range(len(hists[0]))])
   plt.ylabel('Count')
   plt.title('Bins')
   plt.show()
   return hist

# load and augment the data, takes 3-4 minutes
train_df, train_images = load_and_format('train.json')


#getHogDescriptor(train_images[2,:,:,0])
#getHogDescriptor(train_images[5,:,:,0])
#getHogDescriptor(train_images[6,:,:,0])
#getHogDescriptor(train_images[11,:,:,0])
#getHogDescriptor(train_images[1,:,:,0])


### boat histogram overview: 0,1,3,4
##fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
##obj_list = dict(ships = train_df.query('is_iceberg==0').sample(4).index,
##     icebergs = train_df.query('is_iceberg==1').sample(4).index)
##for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
##    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
##    ax1.set_title('%s Band 1' % obj_type)
##    ax1.axis('off')
##    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
##    ax2.set_title('%s Band 2' % obj_type)
##    ax2.axis('off')


# iceberg histogram overview: 2,5,6,10
##fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
##obj_list = dict(ships = train_df.query('is_iceberg==0').sample(4).index,
##     icebergs = train_df.query('is_iceberg==1').sample(4).index)
##for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
##    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
##    ax1.set_title('%s Band 1' % obj_type)
##    ax1.axis('off')
##    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
##    ax2.set_title('%s Band 2' % obj_type)
##    ax2.axis('off')



##image = train_images[100,:,:,0]
##print(statsOfGradients(image))

#def main():
    ####### Hey Atharva and Dhruv here are the augmented data:) #######
#augmented, augmented_y = load_and_augment()
#testaugmented(augmented)
    ###################################################################
##train_df, train_images = load_and_format('train.json')
##print('training', train_df.shape, 'loaded', train_images.shape)
##img = train_images[113,:,:,0]
###img = cv2.imread('dave.jpg',0)
##
##laplacian = cv2.Laplacian(img,cv2.CV_64F)
##sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
##sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
##
##plt.subplot(2,2,1),plt.imshow(img)
##plt.title('Original'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,2),plt.imshow(laplacian)
##plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,3),plt.imshow(sobelx)
##plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,4),plt.imshow(sobely)
##plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
##
##plt.show()


# backup code
    

#def main():
##train_df, train_images = load_and_format('train.json')
##train_y_df, train_y = load_and_format2('train.json')
##print('training', train_df.shape, 'loaded', train_images.shape)
##
##
##### training data overview
####fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
####obj_list = dict(ships = train_df.query('is_iceberg==0').sample(16).index,
####     icebergs = train_df.query('is_iceberg==1').sample(16).index)
####for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
####    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
####    ax1.set_title('%s Band 1' % obj_type)
####    ax1.axis('off')
####    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
####    ax2.set_title('%s Band 2' % obj_type)
####    ax2.axis('off')
##
###showimage(train_images[7,:,:,1],train_images[7,:,:,1],"","")
##
##
###descriptor_vector(train_images[100,:,:,0])
##for i in range(len(train_images)):
##    train_images[i,:,:,0] = modify(train_images[i,:,:,0])
##    train_images[i,:,:,1] = modify(train_images[i,:,:,1])
##### modified data overview
####fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
####obj_list = dict(ships = train_df.query('is_iceberg==0').sample(16).index,
####     icebergs = train_df.query('is_iceberg==1').sample(16).index)
####for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
####    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
####    ax1.set_title('%s Band 1' % obj_type)
####    ax1.axis('off')
####    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
####    ax2.set_title('%s Band 2' % obj_type)
####    ax2.axis('off')
##
##MAXMAG, MAXANG = findmax(train_images)
##X = np.ndarray((len(train_images), 17*4*2))
##
##
##
###for i in range(101,105):
###    print(train_y[i])
##for i in range(len(train_images)):   #(len(train_images)): 
##    v1 = descriptor_vector(train_images[i,:,:,0])
##    v2 = descriptor_vector(train_images[i,:,:,1])
####    if len(np.concatenate([v1,v2])) == 130:
####        print("130", i, len(v1), len(v2))
####    if len(np.concatenate([v1,v2])) == 131:
####        print("131", i, len(v1), len(v2))
##    X[i] = np.concatenate([v1,v2])
##print(X.shape)
###m1 = descriptor_vector(train_images[7,:,:,0])
##
###descriptor_vector(train_images[548,:,:,1])
##
##plt.show()



#descriptor_vector(train_images[50])
#print(descriptor_vector(train_images[1001,:,:,0]) + descriptor_vector(train_images[1001,:,:,1]))
#for i in range(len(train_images)):
#    m1, a1 = descriptor_vector(train_images[i,:,:,0])
#    m2, a2 = descriptor_vector(train_images[i,:,:,1])
#    X[i] = descriptor_vector(train_images[i,:,:,0]) + descriptor_vector(train_images[i,:,:,1])
#print(X.shape)

#main()








    #svm(X,train_y)
        
    #descriptor_vector(train_images[100,:,:,1])
    #hog2(train_images[10,:,:,1])
    #descriptor_vector(modify(train_images[10,:,:,1]))
    #showimage(train_images[100,:,:,1],modify(train_images[100,:,:,1]),"original","modified")
    #print(train_y[10])

    #plt.show()
#main()



##    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
##    ax1.matshow(train_images[1001,:,:,0])
##    ax1.set_title('band 1')
##    ax2.matshow(shift(train_images[1001,:,:,0]))
##    ax2.set_title('band 2')
##
##    plt.show()


    #print("new size",augmented.shape)

##    fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
##    obj_list = dict(ships = train_df.query('is_iceberg==0').sample(16).index,
##         icebergs = train_df.query('is_iceberg==1').sample(16).index)
##    for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
##        ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
##        ax1.set_title('%s Band 1' % obj_type)
##        ax1.axis('off')
##        ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
##        ax2.set_title('%s Band 2' % obj_type)
##        ax2.axis('off')

##    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
##    ax1.matshow(train_images[0,:,:,1])
##    ax1.set_title('before')
##    ax2.matshow(shift(train_images[0,:,:,1]))
##    ax2.set_title('shifted')
##
##    plt.show()

   
    #with open('augmented.json', 'w') as f:
    #    json.dump(augmented.tolist(), f)
    #print("done writing")
    
    #with open('augmented.json', 'r') as g:
    #    data = json.load(g)
    #a = np.array(data)
        
    #print(a.shape)

 

## 
# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
#for angle in np.arange(0, 360, 15):
#    rotated = imutils.rotate_bound(image, angle)
#    cv2.imshow("Rotated (Correct)", rotated)
#3    cv2.waitKey(0)

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
##
### testing data overview
##fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,12))
##idx_list = test_df.sample(49).index
##obj_type = 'Test Data'
##ax1.imshow(montage2d(test_images[idx_list,:,:,0]))
##ax1.set_title('%s Band 1' % obj_type)
##ax1.axis('off')
##ax2.imshow(montage2d(test_images[idx_list,:,:,1]))
##ax2.set_title('%s Band 2' % obj_type)
##ax2.axis('off')

#plt.show()
