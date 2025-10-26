import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import math
import random
from sympy import isprime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def bbin(nr, pad_size=None):
    s = bin(nr)[2:]
    if pad_size:
        s = s.rjust(pad_size, '0')
    ar = list(map(lambda x: int(x), list(s)))
    return ar

max_n = 2**2024
max_n = 100000000
max_size = len(bbin(max_n))
sqr_max_n = math.isqrt(max_n)

random.seed(42)

def rnd_prime(sqr_max_n):
    while True:
        value = random.randrange(sqr_max_n)
        if isprime(value): return value

x = []
y = []
data = []
for p in range(sqr_max_n):
    for q in range(sqr_max_n):
        if isprime(p) and isprime(q):
            n = p*q
            data.append((p,q,n))
            nb = bbin(n, max_size)
            x.append(nb)
            pb = bbin(p, max_size)
            qb = bbin(q, max_size)
            pb.extend(qb)
            y.append(pb)
            if len(x) % 1000 == 0: print(f"loading {len(x)}") 

x = np.array(x)
y = np.array(y)
x = tf.cast(x, dtype=tf.int32)
y = tf.cast(y, dtype=tf.int32)

split_of1 = int(len(x)*0.6)
split_of2 = int(len(x)*0.8)
x_train = x[:split_of1]
y_train = y[:split_of1]
x_val = x[split_of1:split_of2]
y_val = y[split_of1:split_of2]
x_test = x[split_of2:]
y_test = y[split_of2:]
x = y = None

print(f"training set: {len(x_train)} val set: {len(x_val)} test set: {len(x_test)}")

#gfg = tf.data.Dataset.from_tensor_slices((x,y))
#train_ds, test_ds = tf.keras.utils.split_dataset(gfg, left_size=0.8, right_size=0.2)
#train_ds = train_ds.batch(10)
#for batch, (inp, tar) in enumerate(train_ds):
#    print(batch, inp.shape, (None,max_size))

model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(max_size,)),
  tf.keras.layers.Dense(max_size*16, activation='relu'),
  tf.keras.layers.Dense(max_size*16, activation='relu'),
  tf.keras.layers.Dense(max_size*16, activation='relu'),
  tf.keras.layers.Dense(max_size*16, activation='relu'),
  tf.keras.layers.Dense(max_size*2, activation="relu")
])

#loss_fn= tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_fn='mean_squared_error'

# def loss_fn(y_true, y_pred):
#     #y_true = tf.cast(y_true >= 0.5, tf.int32)
#     y_pred = tf.cast(y_pred >= 0.5, tf.int32)
#     res = tf.reduce_all(tf.equal(y_true,y_pred), axis=1)
#     res = tf.math.reduce_sum(tf.cast(res, tf.int32))
#     return res/y_true.shape[-1]
def my_loss(y_true, y_pred):
    threshold = 0.2
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.ones(y_true.shape)
    big_error_loss = tf.zeros(y_true.shape)
    res = tf.where(is_small_error, small_error_loss, big_error_loss)
    res = tf.reduce_all(res, axis=1)
    res = tf.math.reduce_sum(tf.cast(res, tf.int32))
    return res/y_true.shape[-1]

def accurary_fn(y_true, y_pred):
    #y_true = tf.cast(y_true >= 0.5, tf.int32)
    y_pred = tf.cast(y_pred >= 0.5, tf.int32)
    res = tf.reduce_all(tf.equal(y_true,y_pred), axis=1)
    res = tf.math.reduce_sum(tf.cast(res, tf.int32))
    return res/y_true.shape[-1]

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=loss_fn,
              metrics=[accurary_fn])
model.summary()

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

def plot_loss_tf(history):
    fig,ax = plt.subplots(1,1, figsize = (4,3))
    widgvis(fig)
    ax.plot(history.history['binary_accuracy'], label='binary_accuracy', color='r')
    ax.plot(history.history['val_binary_accuracy'], label='val_binary_accuracy', color='g')
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('accuracy (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()

history = model.fit(x_train,y_train, validation_data=(x_val, y_val), epochs=200)
#history = model.fit(train_ds, epochs=5, batch_size=10)
#plot_loss_tf(history)

predict = model(x_test)

test_acc = accurary_fn(y_test, predict)
print(f"test accuracy = {test_acc}")