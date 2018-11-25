import numpy as np
import nn
import q4
from skimage import io

x = np.array([[1,2],[5,4]])
train_x_norm = x-(np.sum(x,axis=0)/x.shape[0])
print(train_x_norm)

