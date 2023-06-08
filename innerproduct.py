# %% [markdown]
# Introduction to inner product and convolution

First: Matrix multiplication
![image.png](attachment:image.png)

Matrices M1 and M2 are multiplied by multiplying the row elements of M1 with the column elements of M2, and the results added for each row/column pair. 
The sign for this is `*`
In Matlab, `*` is matrix multiplication and `.*` is elementwise individual multiplication without any addition. 
In this process, the inner dimensions get lost: 
The product of 2 M1 and M2 exists only if the number of columns of M1 (the second dimension) is equal to the number of rows of M2 (its first dimension).

# %%




# %% [matlab]
a = rand(3,2)
size(a)
b = rand (2,3)
size(b)

# %% [markdown]
The results has dimensions of rows of M1 and columns of M2 (again, the "inner" dimensions are lost, - but not the data of course)

# %%




# %% [matlab]
c = a * b 
size(c)

# %% [markdown]
more examples with randome number matrices. 
what will the new dimensions be? 
rand(4,3) * rand(3,1)

rand(2,1) * rand(1,10)

rand(2,1) * rand(1,1)

# %% [markdown]

[https://timeseriesreasoning.com/contents/deep-dive-into-variance-covariance-matrices/](https://timeseriesreasoning.com/contents/deep-dive-into-variance-covariance-matrices/)

inner product: 
matrix multiplication of two vectors of the same length, one is a row, one is a column vector

# %%




# %% [matlab]
a1 = rand(1,4)
a2 = rand(4,1)

# %% [markdown]
one more time, two vectors

# %%




# %% [matlab]
a1 = [-1 1 -1 1]
figure, bar(a1)
a2 = [-1 -1 1 1]
figure, bar(a2)

# %% [markdown]
The inner product between those two vectors

# %%




# %% [matlab]
c = a1*a2'

# %% [markdown]
so, the inner product is a measure of similarity between two vectors, it is high when they are similar, and low when they are not. In fact, the inner product is like an unscaled correlation coefficient. Let's try this

---

one more time, two vectors

# %%




# %% [matlab]
a1 = [-1 1 -1 1]
figure, bar(a1)
a2 = [-1 .5 -1 1]
figure, bar(a2)

# %% [markdown]
The inner product between those two vectors

# %%




# %% [matlab]
c = a1*a2'
