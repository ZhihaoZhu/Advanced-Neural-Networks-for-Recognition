import numpy as np
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# # returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    gray_img = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(gray_img)
    bw = skimage.morphology.closing(gray_img > thresh, skimage.morphology.square(3))
    bw_copy = bw.copy()
    erosion_selem = np.ones((9, 9), dtype=np.bool)
    processed = skimage.morphology.binary_erosion(bw, selem=erosion_selem)
    thresh_adapt_img = 1 - processed
    label_image, region_num = skimage.measure.label(thresh_adapt_img, connectivity=2, background=0, return_num=True)
    for region in skimage.measure.regionprops(label_image):
        if region.area >= 300:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minr, minc, maxr, maxc])
    bboxes = np.array(bboxes)
    return bboxes, bw_copy

def find_group(bboxes):
    weight = bboxes[:,1]
    height = bboxes[:,2].reshape((1,-1))
    print(height)
    n = height.shape[1]
    index = np.arange(n).reshape((1,-1))
    new_height = np.concatenate((height,index), axis=0)
    print(new_height)

    new_height.sort(axis=1)
    print(new_height)

    return height

def crop(bw, bboxes):
    images = []
    for i in range(bboxes.shape[0]):
        cord = bboxes[i,:]
        image = bw[cord[0]:cord[2], cord[1]:cord[3]]*1
        print(image)

        width, height = image.shape
        diff = abs(width - height) // 2
        if width<height:
            padded_img = np.pad(image, ((diff, diff), (0, 0)), 'constant', constant_values=(1))
        else:
            padded_img = np.pad(image, ((0, 0), (diff, diff)), 'constant', constant_values=(1))
        crop = skimage.transform.resize(padded_img, (32, 32))
        # import matplotlib.pyplot as plt
        # plt.imshow(crop)
        # plt.show()
        # break
        flat = crop.transpose().reshape((1, -1))
        images.append(flat)
    images = np.array(image)

    return images






