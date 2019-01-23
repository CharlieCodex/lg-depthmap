import numpy as np
from PIL import Image
import os

def plist_dir(p):
    return map(lambda x: os.path.join(p, x), os.listdir(p))

if __name__ == '__main__':
    datadir = 'validation/'
    plist = list(plist_dir(datadir))
    arr = np.zeros((len(plist), 400, 400, 4))
    for i, p in enumerate(plist):
        a = np.array(Image.open(p).resize((400,400)))
        arr[i,:,:,:3] = a
    np.save('mid/datasets/validation.npy', arr)