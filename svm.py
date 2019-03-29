# part of the code is from https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 #open cv for image processing, imafe feature extraction
from sklearn import svm # for fitting model to features
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import decomposition
from sklearn import linear_model
from sklearn.metrics import classification_report
from scipy.ndimage.filters import uniform_filter
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import (StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn import datasets
import pdb
import seaborn as sns
from matplotlib import pyplot

sns.set()

'''
Load the data
'''
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Loading the training and testing data
trainDataFilePath = "../input/train.json"
testDataFilePath = "../input/test.json"
trainDataFrame = pd.read_json('../input/train.json')
#trainDataFrame.head()
testDataFrame = pd.read_json('../input/test.json')
#testDataFrame.head()

def decibel_to_linear(band):
     # convert to linear units
    return np.power(10,np.array(band)/10)

def linear_to_decibel(band):
    return 10*np.log10(band)

'''
Denoise
'''
# 1. Use lee filter
# implement the Lee Filter for a band in an image already reshaped into the proper dimensions
def lee_filter(band, window, var_noise = 0.25):
# band: SAR data to be despeckled (already reshaped into image dimensions)
# window: descpeckling filter window (tuple)
# default noise variance = 0.25
# assumes noise mean = 0

     mean_window = uniform_filter(band, window)
     mean_sqr_window = uniform_filter(band**2, window)
     var_window = mean_sqr_window - mean_window**2

     weights = var_window / (var_window + var_noise)
     band_filtered = mean_window + weights*(band - mean_window)
     return band_filtered

# 2. Get rid of the background as much as possible
def modify(image):
    #image = lee_filter(image, 8)
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
    #print(len(copy))
    for i in range(len(copy)):
        for j in range(len(copy[0])):
           for e in copy[i,j]:
              if e < cutoff:
                 e = cutoff
    return copy

# 3. new lee-filter
def E_lee_filter(band, window, var_noise = 0.25,damp = 1, numL = 1):
    # band: SAR data to be despeckled (already reshaped into image dimensions)
    # window: descpeckling filter window (tuple)
    # default noise variance = 0.25
    # assumes noise mean = 0

    mean_window = uniform_filter(band, window)
    mean_sqr_window = uniform_filter(band**2, window)
    var_window = mean_sqr_window - mean_window**2
    SD_window = np.sqrt(var_window);


    C_U = 1/(np.sqrt(numL)*var_noise)
    C_max = np.sqrt(1+2/numL)
    C_L = SD_window/mean_window
    K = np.exp(-damp*(C_L - C_U)/(C_L - C_max))

    if (C_L.all() <= C_U.all()):
        band_filtered = mean_window
    elif (C_L.all()>C_U.all() and C_L < C_max):
        band_filtered = mean_window*K + (1-K)*band
    else:
        band_filtered = band;

    return band_filtered

'''
PCA
'''
def pca_train(train,n):
    hog=[]
    y=[]
    for i in range(train.shape[0]):
        #train["hogFeature"][i]
        hog.append(train["hogFeature"][i])
        y.append(train['is_iceberg'][i])
        
    hog=np.asarray(hog)
    y=np.asarray(y)
    pca = decomposition.PCA(n_components=10)
    pca.fit(hog)
    hog = pca.transform(hog)
    return hog,y,pca

'''
Calculate the Hog features out of the Image
'''
def getHogDescriptor(image,binNumber = 16):
##   image = modify(image)
   image = E_lee_filter(image,8)
   #image = filter2(image)
   gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
   gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
   mag, ang = cv2.cartToPolar(gx, gy)
##   print(mag)
   bins = np.int32(binNumber*ang/(2*np.pi))
# 69 accuracy  
   bin_cells = [bins[20:50,20:50]\
               ,bins[:20, :50], bins[:50, 50:], \
               bins[50:, :50], bins[20:, :20]]
   mag_cells = [mag[20:50,20:50],\
               mag[:20, :50], mag[:50, 50:], \
               mag[50:, :50], mag[20:, :20]]
# 65 accuracy
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
   #print(hist, sum(hist))
##   plt.bar([i+1 for i in range(len(hists[0]))], hists[0], align='center', alpha=0.5)
##   plt.xticks([i+1 for i in range(len(hists[0]))], [i+1 for i in range(len(hists[0]))])
##   plt.ylabel('Count')
##   plt.title('Bins')
##   plt.show()
    

##   plt.hist(hist, bins=16)  # arguments are passed to np.histogram
##   plt.title("Histogram with 'auto' bins")
##   plt.show()
##   #print(len(hist), "hist length")
   return hist

#Get the Colour Composite RGB Image from HH and HV bands
def makeRGBImageFromHnV(bandHH,bandHV):
    b = np.divide(bandHH, bandHV, out=np.zeros_like(bandHH), where=(bandHV!=0))
    rgb = np.dstack((bandHH.astype(np.float32), bandHV.astype(np.float32),b.astype(np.uint16)))
    return rgb

# get mean and std of image dataframe


def getMeanImageFromImageDataFrame(trainDataFrame):
    meanImage = np.zeros(shape =(75,75,3),dtype = np.float32)
    for currentImage in trainDataFrame["fullImage"]:
        meanImage = meanImage + currentImage
    meanImage = meanImage/len(trainDataFrame)
    return meanImage.astype(np.float32)

def getStandardDeviationFromImageDataFrame(trainDataFrame,meanImage):
    stdImage = np.zeros(shape =(75,75,3),dtype = np.float32)
    for currentImage in trainDataFrame["fullImage"]:
        stdImage = stdImage + (currentImage - meanImage)
    stdImage = stdImage/len(trainDataFrame)
    return stdImage.astype(np.float32)

def normalizedImageParamFromDataFrame(trainDataFrame):
    
    meanImageData = getMeanImageFromImageDataFrame(trainDataFrame)
    stdImageData = getStandardDeviationFromImageDataFrame(trainDataFrame,meanImageData)
    #for i in range(0,len(trainDataFrame)):
        #currentImage = trainDataFrame["fullImage"][i]
        #trainDataFrame["fullImageNormalized"][i] = (currentImage - meanImageData)/stdImageData
    return meanImageData,stdImageData

def normalizeSingleImage(currentImage,meanImageData,stdImageData):
    normalizedImage = (currentImage - meanImageData)/stdImageData
    return normalizedImage

def transformImageDataFrame(inputDataFrame,meanImageData,stdImageData):
    inputDataFrame["fullImageNormalized"] = [normalizeSingleImage(inputDataFrame["fullImage"][i],meanImageData,stdImageData) for i in range(0,len(inputDataFrame["fullImage"]))]
    return inputDataFrame

def normalizeImageUsingOpenCV(currentImage):
    norm_image = cv2.normalize(currentImage,currentImage, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

# Convert the dB Band values into normalValues
def getValuesfromDB(bandDB):
    currentband = np.array(bandDB).reshape(75,75)
    actualValue = 10**(currentband/10)
    return actualValue

# Adding required columns for the DataFrame
def getImageFromBandDataFrame(dataFrame):
    dataFrame["valueBand1"] = [getValuesfromDB(dataFrame["band_1"][i]) for i in range(0,len(dataFrame["band_1"]))]
    dataFrame["valueBand2"] = [getValuesfromDB(dataFrame["band_2"][i]) for i in range(0,len(dataFrame["band_2"]))]
    dataFrame["fullImage"] = [makeRGBImageFromHnV(dataFrame["valueBand1"][i],dataFrame["valueBand2"][i]) for i in range(0,len(dataFrame["band_1"]))]
    dataFrame["fullImageNormalized"] = [normalizeImageUsingOpenCV(dataFrame["fullImage"][i]) for i in range(0,len(dataFrame["fullImage"]))]
    #dataFrame["hogFeature"] = [getHogDescriptor(dataFrame["fullImage"][i]) for i in range(0,len(dataFrame["fullImage"]))]
    return dataFrame


def bootstrapAndEqualizeTheData(inputDataFrame):
    noOfIceBergData = len(inputDataFrame[inputDataFrame.is_iceberg == 1])
    totalData = len(inputDataFrame)
    print("Randomly bootstrap and make the iceberg and ship data to be equal")
    randomSamplesForIceBerg = inputDataFrame[inputDataFrame.is_iceberg == 1].sample((totalData - (2*noOfIceBergData)))
    inputDataFrame = pd.concat([inputDataFrame, randomSamplesForIceBerg], ignore_index=True)
    return inputDataFrame


def addFeatureDataFrame(dataFrame):
    #dataFrame["hogFeature"] = [getHogDescriptor(dataFrame["fullImage"][i]) for i in range(0,len(dataFrame["fullImage"]))]
    dataFrame["hogFeature"] = [getHogDescriptor(dataFrame["fullImageNormalized"][i]) for i in range(0,len(dataFrame["fullImageNormalized"]))]
    return dataFrame

# Extracting Features from the Data Frame
def getFeatureFromDataFrame(dataFrame,isTestData=0):
    featureDataVector = []
    responseVector = []
    featureDataVector =  np.array(featureDataVector).reshape(-1,80)
    for i in range(0,len(dataFrame)):
        currentFeature = dataFrame["hogFeature"][i].tolist()
        if(isTestData is 0):
            currentResponse = dataFrame["is_iceberg"][i].tolist()
        else:
            currentResponse = 2 # dummy which will be ignored later
        currentFeature = np.array(currentFeature[0:80]).reshape(-1,80)
        currentResponse = int(currentResponse)
        if(i == 0):
            featureDataVector = currentFeature
            responseVector.append(currentResponse)
        else:
            featureDataVector = np.vstack((featureDataVector,currentFeature))
            responseVector.append(currentResponse)
    return featureDataVector,responseVector

def filter1(img):
   return cv2.Laplacian(img,cv2.CV_64F)

def filter2(img):
   return cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

def filter3(img):
   return cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)


print("Adding New Columns to DataFrame")


print("Normalizing Data Frame")
#scaler = preprocessing.MinMaxScaler()
#pdb.set_trace()
#trainDataFrame["fullImage"] = trainDataFrame["fullImage"].astype(np.float32)
#trainDataFrame["fullImage"] = scaler.fit_transform(trainDataFrame["fullImage"])
#meanImage = int(np.mean(trainDataFrame["fullImage"]))


#meanImageData,stdImageData = normalizedImageParamFromDataFrame(trainDataFrame)
#trainDataFrame = transformImageDataFrame(trainDataFrame,meanImageData,stdImageData)
#print(trainDataFrame["valueBand1"][0])

##aug, aug_y = augment.augmented, augment.augmented_y
##print(trainDataFrame.shape)

##temp = trainDataFrame.copy()
##L = []
##for i in range(1604):
##   L.append(i+1604)
##temp["id"] = L
##trainDataFrame = pd.concat([trainDataFrame,temp])


##tempset = [trainDataFrame.copy(),trainDataFrame.copy(),trainDataFrame.copy()]
##filters = [filter1,filter2,filter3]
##
##temp = tempset[0]
##ids = []
##band1 = []
##band2 = []
##for i in range(1604):
##    img1 = trainDataFrame["valueBand1"][i]
##    img2 = trainDataFrame["valueBand2"][i]
##    ids.append(3*i)
##    ids.append(3*i+1)
##    ids.append(3*i+2)
##    band1.append(filter1(img1))
##    band1.append(filter2(img1))
##    band1.append(filter3(img1))
##    band2.append(filter1(img2))
##    band2.append(filter2(img2))
##    band2.append(filter3(img2))
##    
##tempset[0]["id"] = ids[:1604]
##tempset[1]["id"] = ids[1604:3208]
##tempset[2]["id"] = ids[3208:4812]
##tempset[0]["valueBand1"] = band1[::3]
##tempset[0]["valueBand2"] = band2[::3]
##tempset[1]["valueBand1"] = band1[1::3]
##tempset[1]["valueBand2"] = band2[1::3]
##tempset[2]["valueBand1"] = band1[2::3]
##tempset[2]["valueBand2"] = band2[2::3]
##trainDataFrame = pd.concat([trainDataFrame,tempset[0]])
##trainDataFrame = pd.concat([trainDataFrame,tempset[1]])
##trainDataFrame = pd.concat([trainDataFrame,tempset[2]])
##
##print(trainDataFrame.shape)

##for i in range(2):
##    temp = tempset[i]
##    ids = []
##    band1 = []
##    band2 = []
##    for j in range(1604):
##       img1 = trainDataFrame["valueBand1"][j]
##       img2 = trainDataFrame["valueBand2"][j]
##       ids.append(j+(i+1)*1604)
##       band1.append(filters[i+1](img1))
##       band2.append(filters[i+1](img2))
##    temp["id"] = ids
##    temp["valueBand1"] = band1
##    temp["valueBand2"] = band2
##    trainDataFrame = pd.concat([trainDataFrame,temp])

trainDataFrame = getImageFromBandDataFrame(trainDataFrame)
trainDataFrame = bootstrapAndEqualizeTheData(trainDataFrame)
trainDataFrame = addFeatureDataFrame(trainDataFrame)



#pca_train(trainDataFrame,len(trainDataFrame))

print("Calculating Feature Vectors")
trainFeatureData , trainResponseData = getFeatureFromDataFrame(trainDataFrame)

print("Getting features from test data")
testDataFrame = getImageFromBandDataFrame(testDataFrame)
testDataFrame = addFeatureDataFrame(testDataFrame)
testFeatureData , testResponseData = getFeatureFromDataFrame(testDataFrame, 1)
print("done..")




##newFeatureData = np.zeros((1702, 64*4))
##for i in range(1702):
##   a1 = np.concatenate((trainFeatureData[i], trainFeatureData[i+1604]))
##   a2 = np.concatenate((a1, trainFeatureData[i+1604*2]))
##   a3 = np.concatenate((a2, trainFeatureData[i+1604*3]))
##   newFeatureData[i] = a3
##trainFeatureData = newFeatureData
##trainResponseData = trainResponseData[:1702]


#print("Feature Normalization")
#scaler = preprocessing.StandardScaler().fit(trainFeatureData)
#trainFeatureData = scaler.transform(trainFeatureData)

#scaler = preprocessing.QuantileTransformer(random_state=0)
#trainFeatureData = scaler.fit_transform(trainFeatureData)

#scaler = preprocessing.MinMaxScaler()
#trainFeatureData = scaler.fit_transform(trainFeatureData)

##print(str(trainFeatureData.shape))
#pdb.set_trace()
print("Fitting Data to Model")
#clf = svm.SVC(gamma=0.01,C=10,kernel='poly',probability=True)


#clf = linear_model.Lasso(alpha=0.1)

clf = svm.SVC(gamma=0.001,C=100.0,kernel='rbf',probability=True)

#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),algorithm="SAMME",n_estimators=500)
#clf1 = DecisionTreeClassifier(max_depth=4)
#clf2 = KNeighborsClassifier(n_neighbors=7)
#clf3 = SVC(kernel='rbf', probability=True)
#param_grid = {'kernel':('poly','rbf'),'C':[1,10,100],'gamma':[0.0001,0.01,0.1]}
#clf = GridSearchCV(svm.SVC(probability=True), param_grid)

#clf.fit(trainFeatureData[:1190],trainResponseData[:1190])

clf.fit(trainFeatureData,trainResponseData)
#trainPredictData = clf.predict(trainFeatureData[1190:])
#trainPredictData = [1*((1-e)<e) for e in trainPredictData]
#print(trainPredictData)
trainPredictData = clf.predict(trainFeatureData)
trainPredictData = [1*((1-e)<e) for e in trainPredictData]

print("Evaluating for train data")
#scores=cross_val_score(clf,trainFeatureData[1190:],trainResponseData[1190:],cv=10,scoring='accuracy')

# Cross validation
scores=cross_val_score(clf,trainFeatureData,trainResponseData,cv=10,scoring='accuracy')

# Regular score on training data
##scorelist = [trainPredictData[i]==trainResponseData[i] for i in range(len(trainPredictData))]
##scores = sum(scorelist)/len(trainPredictData)
##scores *= 100
##print(scores)

# Using Lasso model
##Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
##   normalize=False, positive=False, precompute=False, random_state=None,
##   selection='cyclic', tol=0.0001, warm_start=False)
##
#print(clf.score(trainFeatureData,trainResponseData))

print(scores*100)
scores=scores*100
print("Mean of all scores is %f"%(float(float(sum(scores))/10)))

y_pred = cross_val_predict(clf, trainFeatureData, trainResponseData)
#print(y_pred)

##fig, ax = plt.subplots()
##ax.scatter(trainResponseData, , edgecolors=(0, 0, 0))
##ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
##ax.set_xlabel('Measured')
##ax.set_ylabel('Predicted')
##plt.show()



##trainPredictData0 = clf.predict(trainFeatureData)
##trainPredictData = [1*((1-e)<e) for e in trainPredictData0]
y_true = trainResponseData
#y_pred = trainPredictData
#matrix = sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2)
#print(classification_report(y_true, y_pred, digits = 5))

logreg = LogisticRegression()
logreg.fit(trainFeatureData, trainResponseData)
proba = logreg.predict_proba(trainFeatureData)
proba = proba[:,1]

fpr, tpr, thresholds = roc_curve(y_true, proba)
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="upper left")
plt.show()


#print("Feature vector classifier:\n")
#print(classification_report(testResponseData, testPredictData))

##print("Done")
##
##print("getting TestData features...")
##print("Normalizing Data Frame")
##testDataFrame = getImageFromBandDataFrame(testDataFrame)
###testDataFrame = transformImageDataFrame(testDataFrame,meanImageData,stdImageData)
###testDataFrame["fullImage"] = scaler.transform(testDataFrame["fullImage"])
##testDataFrame = addFeatureDataFrame(testDataFrame)
###testDataFrame = getValuesFromDBDataFrame(testDataFrame)
##testFeatureData , _ = getFeatureFromDataFrame(testDataFrame,1)
###testFeatureData , testResponseData = trainFeatureData , trainResponseData
##print("done")
##
###pdb.set_trace()
##
###trainFeatureData = scaler.transform(trainFeatureData)
##print(str(testFeatureData.shape))
##test_predictions = clf.predict_proba(testFeatureData)
##trainPredictions = clf.predict_proba(trainFeatureData)
##print("Prediction done")
##
##
##
##print("Log Loss for training Data is "+str(log_loss(trainResponseData,trainPredictions[:,1])))
##

#pdb.set_trace()

##pred_df = testDataFrame[['id']].copy()
##pred_df['is_iceberg'] = test_predictions[:,1]
##pred_df.to_csv('predictions.csv', index = False)
#pred_df.sample(3)
