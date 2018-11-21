import numpy as np
from nn import *


x = np.ones((2,1))

y = np.ones((1,2))*2

z = x@y
print(z)

b = np.array([1,2])

out = z+b


print(out)