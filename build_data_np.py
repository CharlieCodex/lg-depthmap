import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.misc import imresize
from PIL import Image, ImageFilter
import os.path as path
import os

IMSIZE = (400, 400)
dtype = np.dtype([('img','u1',IMSIZE+(3,)),('depth', 'f4', IMSIZE)])
# np.asarray((img, depth), dtype=dtype).reshape(1)

def preprocess(d):
    d_ = ndimage.maximum_filter(d, size=3, mode='nearest')
    d_ = ndimage.maximum_filter(d_, size=2, mode='nearest')
    return d_

def read_dir(dir_path, scale=False):
    disp0_path = path.join(dir_path, 'disp0.pfm')
    disp1_path = path.join(dir_path, 'disp1.pfm')

    im0_path = path.join(dir_path, 'im0.png')
    im1_path = path.join(dir_path, 'im1.png')
    im1E_path = path.join(dir_path, 'im1E.png')
    im1L_path = path.join(dir_path, 'im1L.png')

    calib_path = path.join(dir_path, 'calib.txt')

    print('Loading disparities')
    disp0 = utils.readPFM(disp0_path)[0]
    disp1 = utils.readPFM(disp1_path)[0]

    print('Loading images')
    im0 = np.array(Image.open(im0_path).convert('RGB'))
    im1 = np.array(Image.open(im1_path).convert('RGB'))
    im1E = np.array(Image.open(im1E_path).convert('RGB'))
    im1L = np.array(Image.open(im1L_path).convert('RGB'))

    if scale:
        print('Scaling Images')
        im0 = imresize(im0, IMSIZE+im0.shape[2:])
        im1 = imresize(im1, IMSIZE+im1.shape[2:])
        im1E = imresize(im1E, IMSIZE+im1E.shape[2:])
        im1L = imresize(im1L, IMSIZE+im1L.shape[2:])

    print('Loading calib')
    calib = utils.read_calib(calib_path)

    b = calib['baseline']
    f = calib['cam0'][0,0]
    d = calib['doffs']

    print('Creating depth maps')

    d0 = b*f / (disp0+d)
    d1 = b*f / (disp1+d)

    d0 = preprocess(d0)
    d1 = preprocess(d1)
    if scale:
        print('Scaling depth maps')
        d0 = imresize(d0, IMSIZE)[..., None]
        d1 = imresize(d1, IMSIZE)[..., None]

    return im0, im1, im1E, im1L, d0, d1

def transform_squares(dir_path):
    im0, im1, im1E, im1L, d0, d1 = read_dir(dir_path)
    w, h = im0.shape[:-1]
    print('Creating windows')
    stride = 50
    # TODO RESIZE IMAGE BEFORE INDEXING !!!!!
    if w > h:
        s = h
        w_, h_ = scale_size = (int(400/h*w), 400)
        print('Creating index')
        idx = np.arange(w_-h_)[None, :]+np.arange(h_)[:, None]
        print('Scaling/Indexing images (4 left)')
        im0 = imresize(im0, scale_size)[idx].transpose((1,2,0,3))
        print('Indexing images (3 left)')
        im1 = imresize(im1, scale_size)[idx].transpose((1,2,0,3))
        print('Indexing images (2 left)')
        im1E = imresize(im1E, scale_size)[idx].transpose((1,2,0,3))
        print('Indexing images (1 left)')
        im1L = imresize(im1L, scale_size)[idx].transpose((1,2,0,3))
        print('Indexing depth maps')
        d0 = imresize(d0, scale_size)[idx, None].transpose((1,2,0,3))
        d1 = imresize(d1, scale_size)[idx, None].transpose((1,2,0,3))
    else:
        s = w
        w_, h_ = scale_size = (400, int(400/w*h))
        print('Creating index')
        idx = np.arange(h_-w_)[None, :]+np.arange(w_)[:, None]
        print('Scaling/Indexing images (4 left)')
        im0 = imresize(im0, scale_size)[:, idx].transpose((2,1,0,3))
        print('Indexing images (3 left)')
        im1 = imresize(im1, scale_size)[:, idx].transpose((2,1,0,3))
        print('Indexing images (2 left)')
        im1E = imresize(im1E, scale_size)[:, idx].transpose((2,1,0,3))
        print('Indexing images (1 left)')
        im1L = imresize(im1L, scale_size)[:, idx].transpose((2,1,0,3))
        print('Indexing depth maps')
        d0 = imresize(d0, scale_size)[:, idx, None].transpose((2,1,0,3))
        d1 = imresize(d1, scale_size)[:, idx, None].transpose((2,1,0,3))

    print(im0.shape)
    print(im1.shape)
    print(d0.shape)
    print(d1.shape)

    return np.array((
        np.concatenate((im0, d0), axis=-1),
        np.concatenate((im1, d1), axis=-1),
        np.concatenate((im1L, d1), axis=-1),
        np.concatenate((im1E, d1), axis=-1),
    )).reshape((-1, 400, 400, 4,))

def transform(dir_path):
    im0, im1, im1E, im1L, d0, d1 = read_dir(dir_path, scale=True)
    return np.array((
        np.concatenate((im0, d0), axis=-1),
        np.concatenate((im1, d1), axis=-1),
        np.concatenate((im1L, d1), axis=-1),
        np.concatenate((im1E, d1), axis=-1),
    ))

def build_squares(data_dir):
    plist = list(plist_dir(data_dir))
    n_scenes = len(plist)
    print('Allocating array')
    saves = 0
    buffsize = 2000
    arr = np.empty(shape=(buffsize, *IMSIZE, 4), dtype='float32')
    n = 0
    print('Allocated')
    for i, p in enumerate(plist):
        print('File {} of {}'.format(i+1, n_scenes))
        print('Doing transform')
        a = transform_squares(p)
        if n + a.shape[0] > buffsize:
            print('Flushing')
            np.save('mid/datasets/squares-{}.npy'.format(saves), arr[:n])
            saves += 1
            n = 0
        
        print('Assigning entry @ pos {} of {}'.format(n, buffsize))
        arr[n:n+a.shape[0], ...] = a
        n += a.shape[0]

    np.save('mid/datasets/squares-{}.npy'.format(saves), arr[:n])

def build_stretched(data_dir):
    plist = list(plist_dir(data_dir))
    n_scenes = len(plist)
    print('Allocating array')
    arr = np.empty(shape=(len(plist)*4, *IMSIZE, 4))
    print('Allocated')
    for i, p in enumerate(plist):
        print('File {} of {}'.format(i+1, n_scenes))
        print('Doing transform')
        a = transform(p)
        print('Assigning entry')
        arr[i:i+4, ...] = a

    np.save('mid/datasets/scaled.npy', arr)

def plist_dir(p):
    return map(lambda x: os.path.join(p, x), os.listdir(p))

if __name__ == '__main__':
    data_dir ='mid/data'
    build_squares(data_dir)
