import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import math
import random
from sympy import isprime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


max = 1000

prods = {}
for i in range(max):
    for j in range(max):
        prod = i * j
        if prod in prods:
            i2, j2 = prods[prod]
            if (i + j) < (i2 + j2):
                prods[prod] = (i, j)
        else:
            prods[prod] = (i, j)

data = []
prime_data = []
for i in range(max*max):
    if i in prods:
        data.append(prods[i])
        if isprime(i):
            prime_data.append(prods[i])
    else:
        break

fig,ax = plt.subplots(1,1, figsize = (4,3))
ax.plot(list(map(lambda x: x[0], data)), list(map(lambda x: x[1], data)), label='data', color='b')
ax.plot(list(map(lambda x: x[0], prime_data)), list(map(lambda x: x[1], prime_data)), label='primes', color='r')
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])
#ax.set_xlabel('Epoch')
#ax.set_ylabel('accuracy (cost)')
ax.legend()
ax.grid(True)
plt.show()
