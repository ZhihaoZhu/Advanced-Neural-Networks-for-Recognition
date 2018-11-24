import numpy as np
import nn
import q4
from skimage import io

image=io.imread('../images/01_list.jpg')
# io.imshow(image)
# io.show()
# print(type(image))


q4.findLetters(image)