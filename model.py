import tensorflow as tf

IMSIZE = (400, 400)

# image in
Xin = tf.placeholder(shape=(None, *IMSIZE, 3), dtype=tf.float64)
# ground truth in
Yin = tf.placeholder(shape=(None, *IMSIZE, 1), dtype=tf.float64)
# step counter, used in checkpoint naming and ordering
global_step = tf.train.create_global_step()

# a convolutional layer that downscales an image with set weights and biasies
downscale = tf.layers.Conv2D(
    filters=8, kernel_size=(4,4),
    strides=(4,4), padding='same',
    name='Downscale8')

# a transpose convoluational layer that upscales an image with set weights and biases
upscale = tf.layers.Conv2DTranspose(
    filters=4, kernel_size=(4,4),
    strides=(4,4), padding='same',
    name='Upscale4')

# First convolutional layer
# Converts from 400 x 400 x 3 to 400 x 400 x 8
c1 = tf.layers.Conv2D(
    filters=8, kernel_size=(2,2),
    strides=(1,1), padding='same',
    name='3To8Space')(Xin)
# Downscales from 400 x 400 x 8 to 100 x 100 x 8
d1 = downscale(c1)
# Downscales from 100 x 100 x 8 to 25 x 25 x 8
d2 = downscale(d1)

# Converts from 25 x 25 x 8 to 25 x 25 x 4
l1 = tf.layers.Conv2D(
    filters=4, kernel_size=(6,6),
    strides=(1,1), padding='same',
    name='8To4Space')(d2)

# Flattens from 25 x 25 x 4 to 2500 x 1
flat = tf.layers.flatten(l1)
# Dense mapping from 2500 x 1 to 2500 x 1
l2 = tf.layers.dense(inputs=flat, units=25*25*4, name="Dense")
# Reshapes from 2500 to 25 x 25 x 4
full = tf.reshape(l2, shape=(-1, 25, 25,4,))

# Upscales from 25 x 25 x 4 to 100 x 100 x 4
u1 = upscale(full)
# Upscales from 100 x 100 x 4 to 400 x 400 x 4
u2 = upscale(u1)

# Second simple convolutional layer
# Converts from 400 x 400 x 4 to 400 x 400 x 4
c2 = tf.layers.Conv2D(
    filters=4, kernel_size=(6,6),
    strides=(1,1), padding='same',
    name='Conv1')(u2)
# Third simple convolutional layer
# Converts from 400 x 400 x 4 to 400 x 400 x 6
c3 = tf.layers.Conv2D(
    filters=6, kernel_size=(3,3),
    strides=(1,1), padding='same',
    name='Conv2')(c2)
# Fourth and final simple convolutional layer
# Converts from 400 x 400 x 6 to 400 x 400 x 1
final = tf.layers.Conv2D(
    filters=1, kernel_size=(6,6),
    strides=(1,1), padding='same',
    name='Final')(c3)

# print(c1.shape)
# print(d1.shape)
# print(d2.shape)
# print(l1.shape)
# print(flat.shape)
# print(l2.shape)
# print(full.shape)
# print(u1.shape)
# print(u2.shape)
# print(final.shape)

# Computes the quadratic loss between final layer and ground truth 
# on a pixel by pixel basis, averaged along batch axis and scalar channel axis
loss_ = tf.reduce_mean((final - Yin) ** 2, name="PixelLoss", axis=(0,3))
# Compute scalar mean loss across all pixels
loss = tf.reduce_mean(loss_, name="MeanLoss")

# Creates an AdamOptimizer specialized for reccurent / convolutional networks
optim = tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=5e-8, # to tinker with
    use_locking=False,
    name='Adam')

# Creates an operation that will used gradient descent to tweak weights and biases
# Also increments global_step
train_step = optim.minimize(loss, global_step=global_step)
