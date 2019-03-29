import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import operator
import numpy as np
import pandas as pd
from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from skimage import data, io, filters
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pdb
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from numpy import linalg as LA
from sklearn import decomposition
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import svm

def sobel_fil(train_images,img_no=0,band=0,both=0):
    if both==1:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        matshow()
        ax1.imshow(filters.sobel(train_images[img_no,:,:,0]))
        ax1.set_title('Band 1')
        ax2.imshow(filters.sobel(train_images[img_no,:,:,1]))
        ax2.set_title('Band 2')
        io.show()
        return
    io.imshow(filters.sobel(train_images[img_no,:,:,band]))
    lab='Band '+str(band)
    io.show()
    return


def load_dataset(path='train.json'):
     train=pd.read_json(path)
     train_images = train.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
     train_images = np.stack(train_images).squeeze()
     i=0
     for x in train_images:
         train_images[i,:,:,0]=train_images[i,:,:,0]-np.min(train_images[i,:,:,0]);
         train_images[i,:,:,1]=train_images[i,:,:,1]-np.min(train_images[i,:,:,1]);
         i=i+1
     return train, train_images

def imageshow(train_images,img_no=0,band=0,both=0):
    if both==1:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
        ax1.matshow(train_images[img_no,:,:,0])
        ax1.set_title('Band 1')
        ax2.matshow(train_images[img_no,:,:,1])
        ax2.set_title('Band 2')
        plt.show()
        return
    plt.matshow(train_images[img_no,:,:,band])
    lab='Band '+str(band)
    plt.title(lab)
    plt.show()
    return


def get_stats(train,label=1):
    train['band_'+str(label)]=[list(np.array(x)-np.min(np.array(x))) for x in train['band_'+str(label)]]
    p1=[]
    p2=[]
    p3=[]
    p4=[]
    p5=[]
    p6=[]
    p7=[]
    p8=[]
    p9=[]
    i=0
    for x in train['band_'+str(label)]:
        p1.append(np.mean(show_thresh(i,0.8,label-1)))
        p2.append(np.mean(show_thresh(i,0.7,label-1)))
        p3.append(np.mean(show_thresh(i,0.9,label-1)))
        p4.append(np.mean(show_thresh(i,0.8,0))-np.mean(show_thresh(i,0.8,1)))
        p5.append(np.mean(show_thresh(i,0.7,0))-np.mean(show_thresh(i,0.7,1)))
        p6.append(np.mean(show_thresh(i,0.9,0))-np.mean(show_thresh(i,0.9,1)))
        if label==1:
            p7.append(LA.norm(band1[i]))
            p8.append(LA.norm(band1[i][1]))
            p9.append(LA.norm(band1[i][2]))
        if label==2:
            p7.append(LA.norm(band2[i]))
            p8.append(LA.norm(band2[i][1]))
            p9.append(LA.norm(band2[i][2]))
        i=i+1
    train['thresh0.8_'+str(label)]=p1
    train['thresh0.7_'+str(label)]=p2
    train['thresh0.9_'+str(label)]=p3
    train['thresh0.8_diff']=p4
    train['thresh0.7_diff']=p5
    train['thresh0.9_diff']=p6
    train['pcan_'+str(label)]=p7
    train['pca1_'+str(label)]=p8
    train['pca2_'+str(label)]=p9
    train['max'+str(label)] = [np.max(np.array(x)) for x in train['band_'+str(label)] ]
    train['maxpos'+str(label)] = [np.argmax(np.array(x)) for x in train['band_'+str(label)] ]
    train['min'+str(label)] = [np.min(np.array(x)) for x in train['band_'+str(label)] ]
    train['minpos'+str(label)] = [np.argmin(np.array(x)) for x in train['band_'+str(label)] ]
    train['med'+str(label)] = [np.median(np.array(x)) for x in train['band_'+str(label)] ]
    train['std'+str(label)] = [np.std(np.array(x)) for x in train['band_'+str(label)] ]
    train['mean'+str(label)] = [np.mean(np.array(x)) for x in train['band_'+str(label)] ]
    train['p25_'+str(label)] = [np.sort(np.array(x))[int(0.25*75*75)] for x in train['band_'+str(label)] ]
    train['p75_'+str(label)] = [np.sort(np.array(x))[int(0.75*75*75)] for x in train['band_'+str(label)] ]
    train['mid50_'+str(label)] = train['p75_'+str(label)]-train['p25_'+str(label)]
    train['kurt'+str(label)] = [kurtosis(x) for x in train['band_'+str(label)]]
    train['skew'+str(label)] = [skew(x) for x in train['band_'+str(label)]]
    train['mean'] =list(map(operator.sub, [np.mean(np.array(x)) for x in train['band_1']], [np.mean(np.array(x)) for x in train['band_2']]))
    train['merati_'+str(label)]=[x/y for x, y in zip(train['mean'], train['mean'+str(label)])]
    return train

def thresh(ent,rh=0.9,lim=0):
	if lim>0:
		binary=ent>lim
		return binary
	thresh=ent.max()*rh
	binary=ent>thresh
	return binary

def show_thresh(i,n=0.8,channel=0,disp=0):
    ceil=np.max(train_images[i,:,:,0])*n
    #plt.imshow(train_images[i,:,:,channel]*thresh(train_images[i,:,:,channel],n,ceil))
    #plt.show()
    return thresh(train_images[i,:,:,channel],n,ceil)

def plot_var(name,nbins=50):
    minval = train[name].min()
    maxval = train[name].max()
    plt.hist(train.loc[train.is_iceberg==1,name],range=[minval,maxval],
             bins=nbins,color='b',alpha=0.5,label='Boat')
    plt.hist(train.loc[train.is_iceberg==0,name],range=[minval,maxval],
             bins=nbins,color='r',alpha=0.5,label='Iceberg')
    plt.legend()
    plt.xlim([minval,maxval])
    plt.xlabel(name)
    plt.ylabel('Number')
    plt.show()

def corr_mat(train):
    train_stats = train.drop(['id','band_1','band_2'],axis=1)
    corr = train_stats.corr()
    fig = plt.figure(1, figsize=(10,10))
    plt.imshow(corr,cmap='inferno')
    labels = np.arange(len(train_stats.columns))
    plt.xticks(labels,train_stats.columns,rotation=90)
    plt.yticks(labels,train_stats.columns)
    plt.title('Correlation Matrix of Global Variables')
    cbar = plt.colorbar(shrink=0.85,pad=0.02)
    plt.show()

def create_set(train_images,train):
    images=[]
    y=[]
    for i in range(train_images.shape[0]):
        b1=train_images[i,:,:,0]
        b2=train_images[i,:,:,1]
        b3=b1/b2
        r=(b1-b1.min())/(b1.max()-b1.min())
        g=(b2-b2.min())/(b2.max()-b2.min())
        b=(b3-b3.min())/(b3.max()-b3.min())
        y.append(train['is_iceberg'][i])
        final = np.dstack((r, g, b))
        images.append(final)
    y = to_categorical(train.is_iceberg.values,num_classes=2)
    images=np.asarray(images)
    return images,y

def pca_train(train,n):
    band1=[]
    band2=[]
    y=[]
    for i in range(train.shape[0]):
        band1.append(train['band_1'][i])
        band2.append(train['band_2'][i])
        y.append(train['is_iceberg'][i])
    band1=np.asarray(band1)
    band2=np.asarray(band2)
    y=np.asarray(y)
    pca = decomposition.PCA(n_components=n)
    pca.fit(band1)
    band1 = pca.transform(band1)
    pca = decomposition.PCA(n_components=n)
    pca.fit(band2)
    band2 = pca.transform(band2)
    return band1,band2,y,pca

def push_train(train):
    band1=[]
    band2=[]
    y=[]
    for i in range(train.shape[0]):
        band1.append(train['band_1'][i])
        band2.append(train['band_2'][i])
        y.append(train['is_iceberg'][i])
    band1=np.asarray(band1)
    band2=np.asarray(band2)
    y=np.asarray(y)
    return band1,band2,y


def covar_f(train):
    band1=[]
    band2=[]
    y=[]
    for i in range(train.shape[0]):
        band1.append(train['band_1'][i])
        band2.append(train['band_2'][i])
        y.append(train['is_iceberg'][i])
    band1=np.asarray(band1)
    band2=np.asarray(band2)
    y=np.asarray(y)
    c1=np.cov(band1)
    c2=np.cov(band2)
    return c1,c2

def side_plot(n):
    a=train_images[n,:,:,0]-np.min(train_images[n,:,:,0]);
    b=train_images[n,:,:,1]-np.min(train_images[n,:,:,1]);
    plt.imshow(np.hstack((a,b)))
    plt.show()
train, train_images=load_dataset()
band1,band2,y,pca=pca_train(train,50)
train = get_stats(train,1)
train = get_stats(train,2)
X,y=create_set(train_images,train)
Xtr, Xv, ytr, yv = train_test_split(X, y, shuffle=False, test_size=0.20)

#train['inc_angle'] = pd.to_numeric(train['inc_angle'],errors='coerce')
#train_set=create_set(train_images)
#for col in ['inc_angle','min1','max1','std1','med1','mean1','mid50_1']:
    #plot_var(col)
