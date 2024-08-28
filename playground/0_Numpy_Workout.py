# %% [markdown]
"""
# Numpy Workout

A short remainder on `numpy` with exercises

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Ghislain Durif (CNRS, LBMC)  
Monday, August 28, 2023

# Starting slowly

Short application about `numpy`, just a refresh

## Some loops
"""

# %%
import numpy as np

a = np.random.rand(100)
b = np.random.rand(100)

# %% [markdown]
"""
## Dot product

Exercises:

1.  numpy dot product of a and b
2.  write manual loop
3.  compare results



## Matrix product

Exercice:

1.  Generate 2 random matrix $5 \times 5$
2.  Numpy matrix product of both
3.  Manual loop, compare result
4.  Manual loop with numpy dot product of row-column, conpare resurt

Use `np.testing` with the right assert function for comparison purpose.
"""
#%%

res1 = np.dot(a,b) 
#%%
result=0
for i in range(len(a)):
    result += a[i]*b[i]
res2 = result     

#%%
np.testing.assert_array_almost_equal(res1,res2)
#%%
matrix_a = np.random.rand(5,5)
matrix_b = np.random.rand(5,5)

#%%
matrix_prod = np.dot(matrix_a,matrix_b)
matrix_prod
#%%
matrix = np.zeros((5, 5))
for i in range(matrix_a.shape[0]):
    for j in range(matrix_b.shape[1]):
        for k in range(matrix_a.shape[1]):
            matrix[i, j] += matrix_a[i, k] * matrix_b[k, j]

matrix
#%%
np.testing.assert_array_equal(matrix_prod,matrix)

#%%
np.testing.assert_array_almost_equal(matrix_prod,matrix)

#%%
matrix = np.zeros((5, 5))
for i in range(matrix_a.shape[0]):
    for j in range(matrix_b.shape[1]):
              matrix[i, j] = np.dot(matrix_a[i], matrix_b[:,j])         
matrix
#%%


"""
## Colum-wise sum

"""
# %%
D = np.arange(10).reshape(2,5)
D

# %% [markdown]
"""
Exercise:

1.  Col-wise sum of D with numpy
2.  the same with loop(s), compare results

# Ramping up, sequence searching

A frequent in image processing is to find a precise pattern in a given
image. We will restrict ourselves to the one-dimensional case (a list of
positive integers), and we will first try to implement this algorithm.

We want a function that takes as argument two unidimensional numpy
arrays, the first contains the data, and the second the sequence we want
to find in the data. The function returns the list of indices in the
array of data, indices which correspond to the start of each subsequence
of data identical to the sequence we are searching for.

## Illustration

<figure>
<img src="attachment:tikz-figures/sequence-search.svg"
alt="Sequence search" />
<figcaption aria-hidden="true">Sequence search</figcaption>
</figure>

## First blocks
"""
#%%

col_sum = np.sum(D, axis=0)
print(col_sum.shape)
col_sum

# %%

col_sum2 = np.zeros(D.shape[1])
for j in range(D.shape[1]) :
    for i in range(D.shape[0]) :
        col_sum2[j] += D[i, j]

np.testing.assert_array_equal(col_sum, col_sum2)

# %%
data = np.array([1,3,2,0,1,9,2,0,1,1,0],dtype=np.uint8)
sequence = np.array([0,1],dtype=np.uint8)

#%%
seq_size = sequence.shape[0]
seq_ind = np.arange(seq_size)
seq_ind

#%%
data_size = data.shape[0]
cor_size = data_size-seq_size+1
data_ind = np.arange(cor_size).reshape(cor_size,1)
data_ind

#%%
ind_seq = data_ind + seq_ind
ind_seq

#%%
data[ind_seq]

#%%
data[ind_seq] == sequence

#%%
np.all(data[ind_seq] == sequence, axis=1)

#%%
np.argwhere(np.all(data[ind_seq] == sequence, axis=1)).flatten()
# %% [markdown]
"""
We want to get

``` python
numpy_search_sequence(data,séquence)

```

``` python
array([3, 7], dtype=int64)
```

## Exercise

First create an increasing list of indices avec `np.arange` with the
same size of the search sequence

``` python
seq_size = 
seq_ind = 
```

Call `data_size` the size of the input data.

We now want a list of increasing indices from 0 to `data_size-seq_size`,
but transformed into a column vector thanks to `reshape` of numpy. Call
`data_ind` this column vector (dimesion `(data_size-seq_size+1,1)`).

``` python
data_size =
cor_size = data_size-seq_size+1
data_ind = 
```

## Broadcasting

We will then use numpy’s *broadcasting* rules to create a vector of
dimension `(data_size-seq_size+1,2)` which contains the list of all
possible adjacent sequences of indices that we want to locate in the
data as follows:

``` python
array([[ 0,  1],
       [ 1,  2],
       [ 2,  3],
       [ 3,  4],
       [ 4,  5],
       [ 5,  6],
       [ 6,  7],
       [ 7,  8],
       [ 8,  9],
       [ 9, 10]])
```

What very simple operation to perform on `data_ind` and `seq_ind` to get
this?

— [official numpy
doc](https://numpy.org/doc/stable/user/basics.broadcasting.html)

— [french
explanation](https://nbhosting.inria.fr/builds/ue12-python-numerique/handouts/latest/2-05-numpy-broadcast.html)

(10,1) and (2,) are compatible because the first array `data_ind`
contains an unit dimension on the right, the *broadcasting*, it will
first “stretch” on this dimension to match that of `seq_ind`, then
*broadcast* the addition of `seq_ind` over the ten lines, the first
element of `ind_seq` being added to the first element of a line, then
the second element of `seq_ind` adding to the second element of the
corresponding row. The operation is thus repeated on each line.

## Indice expression

Using the result of the previous question as indices for the array of
`data`, apply the search for sequences that are correctly *matched* with
a simple operator. Explain why the result has the same dimension
(*shape*) as `data_ind` and not as `data`.

``` python
array([[False, False],
       [False, False],
       [False, False],
       [ True,  True],
       [False, False],
       [False, False],
       [False, False],
       [ True,  True],
       [False,  True],
       [False, False]])
```

It is the array `data_ind + seq_ind` which is used to *index* the array
of `data`, the corresponding indices are simply used on `data` to
provide the result. Next, numpy performs a *broadcasting* with the `==`
operator which returns a boolean for each element *broadcasted* from
both sides (as before, first element of each line of the first operand
on first element of the sequence to match, etc.)

## Final match

Now we are looking for all lines having a perfect match, ie only `True`.
Use the `np.all` function for this

``` python
array([False, False, False,  True, False, False, False,  True, False,
       False])
```

## Indice extraction

Finally we now extract the indices where there is “match” thanks to
`np.nonzero`

``` python
array([3, 7], dtype=int64)
```
"""
