from model import Xin, Yin, loss, train_step, global_step
from utils import minibatcher
import numpy as np
import tensorflow as tf
import sys
import os
import time

timestamp = int(time.time())

data = np.load('mid/datasets/squares-0.npy')

savedir = 'checkpoints-3/'

if not os.path.exists(savedir):
    os.mkdir(savedir)

if not '--load' in sys.argv:
    load_path = tf.train.latest_checkpoint(savedir)
else:
    idx = sys.argv.index('--load')
    load_path = sys.argv[idx+1]

sess = tf.Session()
saver = tf.train.Saver()

if load_path and not '--new' in sys.argv:
    print('Loading model from {}'.format(load_path))
    saver.restore(sess, save_path=load_path)
else:
    print('Creating new model')
    sess.run(tf.global_variables_initializer())  

batchsize = 16

for epoch, x, y in minibatcher(data, batchsize=batchsize, epochs=1000):
    feed_dict = {
        Xin: x,
        Yin: y
    }
    _, l = sess.run([train_step, loss], feed_dict=feed_dict)
    step = tf.train.global_step(sess, global_step)
    print('Frame: {}/{} (e/s)\tLoss: {}'.format(epoch, step, l))
    if step % 10 == 0:
        save_path = os.path.join(savedir, 'save-{}'.format(timestamp))
        save_path = saver.save(sess, save_path, global_step=global_step)
        print('Saved to... {}'.format(save_path))
