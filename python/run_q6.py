import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32

# Normalize the train_x
train_x_norm = train_x-(np.sum(train_x,axis=0)/train_x.shape[0])
valid_x_norm = valid_x-(np.sum(valid_x,axis=0)/valid_x.shape[0])

#Perform SVD & do PCA

covaraince = train_x_norm.T @ train_x_norm
U,S,Vt = np.linalg.svd(covaraince)

#Calculate the projection matrix
projection = U[:,:dim]

# rebuild a low-rank version
lrank = train_x @ projection
print(lrank.shape)

# rebuild it

recon = lrank @ projection.T
print(recon.shape)


for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(train_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T)
    plt.show()

# build valid dataset
lrank_valid = valid_x @ projection

# rebuild it
recon_valid = lrank_valid @ projection.T

# Calculate the PSNR
total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())