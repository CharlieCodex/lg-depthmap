import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image, ImageFilter

arr1 = utils.readPFM('mid/Adirondack-imperfect/Adirondack-imperfect/disp0.pfm')[0]
arr2 = utils.readPFM('mid/Adirondack-imperfect/Adirondack-imperfect/disp0y.pfm')[0]
arr3 = utils.readPFM('mid/Adirondack-imperfect/Adirondack-imperfect/disp1.pfm')[0]
arr4 = utils.readPFM('mid/Adirondack-imperfect/Adirondack-imperfect/disp1y.pfm')[0]
calib = utils.read_calib('mid/Adirondack-imperfect/Adirondack-imperfect/calib.txt')

b = calib['baseline']
f = calib['cam0'][0,0]
d = calib['doffs']

d1 = b*f / (arr1+d)
d2 = b*f / (arr2+d)
d3 = b*f / (arr3+d)
d4 = b*f / (arr4+d)

d = (d1 - d1.min())/(d1.max()-d1.min())

d_ = ndimage.maximum_filter(d, size=3, mode='nearest')
d_ = ndimage.maximum_filter(d_, size=2, mode='nearest')

fig = plt.figure()

a = fig.subplots(nrows=2, ncols=2).ravel()

a[0].imshow(d1)
a[1].imshow(d2)
a[2].imshow(d3)
a[3].imshow(d4)

plt.show()
