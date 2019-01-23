import tensorflow as tf

IMSIZE = (400, 400)

Xin = tf.placeholder(shape=(None, *IMSIZE, 3), dtype=tf.float64)
Yin = tf.placeholder(shape=(None, *IMSIZE, 1), dtype=tf.float64)
global_step = tf.train.create_global_step()

downscale = tf.layers.Conv2D(
    filters=8, kernel_size=(4,4),
    strides=(4,4), padding='same',
    name='Downscale8')
upscale = tf.layers.Conv2DTranspose(
    filters=4, kernel_size=(4,4),
    strides=(4,4), padding='same',
    name='Upscale4')

c1 = tf.layers.Conv2D(
    filters=8, kernel_size=(2,2),
    strides=(1,1), padding='same',
    name='3To8Space')(Xin)
d1 = downscale(c1)
d2 = downscale(d1)

l1 = tf.layers.Conv2D(
    filters=4, kernel_size=(6,6),
    strides=(1,1), padding='same',
    name='8To4Space')(d2)

flat = tf.layers.flatten(l1)
l2 = tf.layers.dense(inputs=flat, units=25*25*4, name="Dense")
full = tf.reshape(l2, shape=(-1, 25, 25,4,))

u1 = upscale(full)
u2 = upscale(u1)

c2 = tf.layers.Conv2D(
    filters=4, kernel_size=(6,6),
    strides=(1,1), padding='same',
    name='Conv1')(u2)
c3 = tf.layers.Conv2D(
    filters=6, kernel_size=(3,3),
    strides=(1,1), padding='same',
    name='Conv2')(c2)
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

loss_ = tf.reduce_mean((final - Yin) ** 2, name="PixelLoss", axis=(0,3))
loss = tf.reduce_mean(loss_, name="MeanLoss")

optim = tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=5e-8, # to tinker with
    use_locking=False,
    name='Adam')

train_step = optim.minimize(loss, global_step=global_step)