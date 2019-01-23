from model import Xin, Yin, final, loss_
from utils import minibatcher
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from time import time
import os

timestamp = int(time())

data = np.load('mid/datasets/scaled.npy')

savedir = 'checkpoints-2/'

latest = tf.train.latest_checkpoint(savedir)

sess = tf.Session()
saver = tf.train.Saver()

# latest = 'checkpoints/save-1548083585-170'

if latest:
    print('loading model from {}'.format(latest))
    saver.restore(sess, save_path=latest)
else:
    print('Error no checkpoints found')

def sampler(batchsize = 10):
    '''Generator for image input, ground truth, sample image, and discrepancy 4-tuples'''
    for epoch, x, y in minibatcher(data, batchsize=batchsize, epochs=1):
        feed_dict = {
            Xin: x,
            Yin: y,
        }
        f, pl = sess.run([final, loss_], feed_dict=feed_dict)
        yield x, y, f, pl

def slideshow():
    '''Cycle through samples and display them one at a time in a window'''
    for x,y,f,pl in sampler():
        for i in range(x.shape[0]):
            fig, a = plt.subplots(nrows=2, ncols=2)
            a1, a2, a3, a4 = a.ravel()
            a1.imshow(x[i,...])
            a2.imshow(y[i,...,0])
            a3.imshow(f[i,...,0])
            a4.imshow(pl[...])
            plt.show()

def build_sample_deck():
    '''Populates samples/ folder with images containing in their four quandrants  
    Source | Ground Truth  
    Predicted | Discrepancy'''
    n = 0
    out = 'samples/'
    for x,y,f,pl in sampler(batchsize=1):
        a = x[0,...]
        print(a.dtype, a.shape)
        im1 = Image.fromarray(x[0,...].astype('uint8')).convert('RGB')
        im2 = Image.fromarray(y[0,...,0]).convert('RGB')
        im3 = Image.fromarray(f[0,...,0]).convert('RGB')
        im4 = Image.fromarray(pl[...]).convert('RGB')
        w, h = im1.size
        comp = Image.new('RGB', (w*2, h*2))
        comp.paste(im1, (0, 0))
        comp.paste(im2, (w, 0))
        comp.paste(im3, (0, h))
        comp.paste(im4, (w, h))
        comp.save(os.path.join(out, '{}-{}.png'.format(timestamp, n)))
        n+=1

if __name__ == '__main__':
    x = np.load('mid/datasets/validation.npy')
    feed_dict = {
        Xin: x[...,:-1],
    }
    f = sess.run(final, feed_dict=feed_dict)
    w, h = x.shape[1:3]
    for i in range(x.shape[0]):
        im1 = Image.fromarray(x[i,:,:,:].astype('uint8'))
        im2 = Image.fromarray(f[i,:,:,0].astype('uint8'))
        comp = Image.new('RGB', (w, h*2))
        comp.paste(im1, (0,0))
        comp.paste(im2, (0,h))
        comp.save('validation_samples/{}.png'.format(i))
