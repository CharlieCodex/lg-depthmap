import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import PIL

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.filters as filters
import os

sample_dir = '/Users/charliemcvicker/Documents/BlueDodo/VOCdevkit/VOC2012/JPEGImages/'

img_path = '{}{}'.format(sample_dir,os.listdir(sample_dir)[0])

im = Image.open(img_path).convert('L')
sizes = ((500, 500), (100,100), (40, 40))
images = [im.resize(s, Image.LANCZOS) for s in sizes]
arrs = [np.array(image) for image in images]

def verbose_watershed(img):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    selem = np.ones((30,30))
    mean = filters.rank.otsu(img, selem)
    thresh = img < mean
    distance = ndi.distance_transform_edt(thresh)
    print('Did distance')
    local_maxi = peak_local_max(distance, min_distance=1, indices=False, footprint=np.ones((3, 3)),
                                labels=thresh)

    print('Did local maxi')

    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=thresh)

    print('Did watershed')
    return thresh, -distance, labels

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
print(axes)
input()
tpls = [(axes[i],) + verbose_watershed(arrs[i]) for i in range(3)]
for a, i, d, l in tpls:
    a[0].imshow(i, cmap=plt.cm.gray, interpolation='nearest')
    a[0].set_title('Overlapping objects')
    a[1].imshow(d, cmap=plt.cm.gray, interpolation='nearest')
    a[1].set_title('Distances')
    a[2].imshow(l, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    a[2].set_title('Separated objects')

for a in axes.ravel():
    a.set_axis_off()

fig.tight_layout()
plt.show()