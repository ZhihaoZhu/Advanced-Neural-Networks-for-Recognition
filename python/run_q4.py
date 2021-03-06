import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
# np.set_printoptions(threshold=np.inf)


from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    '''
        TODO
    '''
    # group_line = find_group(bboxes)


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    image_set = crop(bw, bboxes)

    # load the weights
    # run the crops through your neural network and print them out

    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    print(image_set.shape)
    # print(image_set[0].reshape((32,32)))


    h1 = forward(image_set, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    print(probs.shape)

    probs_label = np.argmax(probs, axis=1)
    print(probs_label.shape)
    print(probs_label)

    for i in range(5):
        index = i
        image_test = image_set[index, :].reshape((32, 32))
        plt.imshow(image_test.T)
        plt.show()
        print(letters[probs_label[index]])








