
# %% [matlab]
# %% [markdown]
# Introduction to inner product and convolution

# First: Matrix multiplication
# ![image.png](attachment:image.png)

# Matrices M1 and M2 are multiplied by multiplying the row elements of M1 with the column elements of M2, and the results added for each row/column pair. 
# The sign for this is `*`
# In Matlab, `*` is matrix multiplication and `.*` is elementwise individual multiplication without any addition. 
# In this process, the inner dimensions get lost: 
# The product of 2 M1 and M2 exists only if the number of columns of M1 (the second dimension) is equal to the number of rows of M2 (its first dimension).

# %% [markdown]

#[Numpy Inner function](https://numpy.org/doc/stable/reference/generated/numpy.inner.html) :bowtie:
# np.inner(a, b) = sum(a[:]*b[:])

# NOTE that the inner product is not commutative: :smiley:
#[Additional MathIsFun ref suggested from Chat GPT](https://www.mathsisfun.com/algebra/matrix-multiplying.html)

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, filtfilt


# %%

a=np.random.rand(3,2)
print(a.shape)

b=np.random.rand(2,3)
print(b.shape)


# %% [matlab]
# a = rand(3,2)
# size(a)
# b = rand (2,3)
# size(b)

# %% [markdown]
# The results has dimensions of rows of M1 and columns of M2 (again, the "inner" dimensions are lost, - but not the data of course)

# %%

c = np.inner(a, b.T)
print(c.shape)
print(c)


# %% [matlab]
# c = a * b 
# size(c)


# %% [markdown]
# more examples with randome number matrices. 
# what will the new dimensions be? 

# - rand(4,3) * rand(3,1)

# - rand(2,1) * rand(1,10)

# - rand(2,1) * rand(1,1)



# %%

# rand(4,3) * rand(3,1)
print(np.shape(np.inner(np.random.rand(4,3), np.random.rand(3,1).T)))

# rand(2,1) * rand(1,10)
print(np.shape(np.inner(np.random.rand(2,1), np.random.rand(10,1))))

# rand(2,1) * rand(1,1)
print(np.shape(np.inner(np.random.rand(2,1), np.random.rand(1,1).T)))

# %% [markdown]
# A Note :smiley: on the inner product is not commutative (the order matters)

# %%
a1 = np.random.rand(1,4)
a2 = np.random.rand(4,1)

assert np.shape(a1) == np.shape(a2.T)
assert np.inner(a1,a2.T)

# %% [markdown]
# this inequality doesn't return an error :bowtie: 

# %%

if np.inner(a1,a2.T).all != np.inner(a2,a1.T).all:
    print('Not equal (as expected) ')
else: 
    print('Equal')

# %%

assert np.inner(a1,a2.T).all == np.inner(a2,a1.T).all, "Um, not equal!"


# %% [matlab]
# rand(4,3) * rand(3,1)

# rand(2,1) * rand(1,10)

# rand(2,1) * rand(1,1)

# %% [markdown]

#[https://timeseriesreasoning.com/contents/deep-dive-into-variance-covariance-matrices/](https://timeseriesreasoning.com/contents/deep-dive-into-variance-covariance-matrices/)

# inner product: 
# - matrix multiplication of two vectors of the same length, one is a row, one is a column vector

# %%

a1 = np.random.rand(1,4)
a2 = np.random.rand(4,1)


# %% [matlab]
# a1 = rand(1,4)
# a2 = rand(4,1)

# %% [markdown]
# one more time, two vectors

# %%
a1 = np.array([-1, 1, -1, 1])

plt.figure()
plt.bar(range(4),a1)

a2 = np.array([-1, -1, 1, 1])

plt.figure()
plt.bar(range(4),a2)


# %% [matlab]
# a1 = [-1 1 -1 1]
# figure, bar(a1)
# a2 = [-1 -1 1 1]
# figure, bar(a2)

# %% [markdown]
# The inner product between those two vectors

# %%

c = np.inner(a1, a2.T)
print(c)

# %% [matlab]
# c = a1*a2'

# %% [markdown]
#so, the inner product is a measure of similarity between two vectors, it is high when they are similar, and low when they are not. In fact, the inner product is like an unscaled correlation coefficient. Let's try this

#---
# %% [markdown]
# one more time, two vectors

# %%

a1 = np.array([-1,1,-1,1])

plt.figure()
plt.bar(range(4),a1)

a2 = np.array([-1,.5,-1,1])

plt.figure()
plt.bar(range(4),a2)


# %% [matlab]
# a1 = [-1 1 -1 1]
# figure, bar(a1)
# a2 = [-1 .5 -1 1]
# figure, bar(a2)

# %% [markdown]
# The inner product between those two vectors

# %%

c= np.inner(a1, a2.T)
print(c)

# %% [matlab]
# c = a1*a2'

# %% [markdown]
#so, the inner product is a measure of similarity between two vectors, it is high when they are similar, and low when they are not. In fact, the inner product is like an unscaled correlation coefficient. Let's try this


#--
