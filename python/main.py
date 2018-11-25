import numpy as np
import nn
import q4
from skimage import io

import pickle
import string

letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
print(letters)