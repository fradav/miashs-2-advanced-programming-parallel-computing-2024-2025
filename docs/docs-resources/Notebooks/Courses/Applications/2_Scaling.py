# %% [markdown]
"""
# Scaling App with `multiprocessing`

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Ghislain Durif (CNRS, LBMC)  
Monday, August 26, 2024

## Introduction

Let’s say we got a random image and we have to process them mean
pooling.
"""

# %%
import numpy as np

conv_output = np.array([
    [10, 12,  8,  7],
    [ 4, 11,  5,  9],
    [18, 13,  7,  7],
    [ 3, 15,  2,  2]
])
conv_output

# %% [markdown]
"""
## Mean Pooling

We have a 4x4 image and we want to apply a 2x2 mean pooling with a
stride of 2. The output will be a 2x2 image.

First, we extract the 2x2 windows with a stride of 2. We can use the
following function to do so.
"""


# %%
def get_pools_strides(img: np.array, pool_size: int, stride: int) -> np.array:
    # Get the shape of the image
    shape = img.shape
    
    # Get the shape of the output
    out_shape = (
        (shape[0] - pool_size) // stride + 1,
        (shape[1] - pool_size) // stride + 1,
        pool_size,
        pool_size,
    )
    
    # Get the strides of the output
    strides = (
        img.strides[0] * stride,
        img.strides[1] * stride,
        img.strides[0],
        img.strides[1],
    )
    
    # Return the array
    return np.lib.stride_tricks.as_strided(
        img,
        shape=out_shape,
        strides=strides
    )


# %% [markdown]
"""
We check that the function works as expected.

Now, we want to compute the mean of each 2x2 window with
[`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html).

## Reference run (single core)

Time the execution of the function with a single core on a 10000x10000
image.

# Strong scaling

What should we do with the image to speed up the computation? We can
split the strides into chunks and compute the mean of each chunk in
parallel.

[`np.array_split`](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html)
can be used to split the array into chunks, and
[`np.vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html)
to stack the results.

Try it with a 10000x10000 image, and check that the result is the same
as the reference run.

Now let’s multithread it. Just use
[`multiprocessing.pool.ThreadPool`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.ThreadPool)
to parallelize the computation, and `pool.map` to apply the function to
each chunk (no `asyncio` here, we want to wait for the result). The
function should get the image, the number of chunks, and the number of
cores as arguments.

Check that the result is the same as the reference run and time the
execution.
"""

# %%
import functools
from multiprocessing.pool import ThreadPool
 
def meanpool(a):
    . . .

def strong_scaling_main(img,ncores):
    . . .


N=10000
img=np.arange(N*N).reshape(N,N)
res_thread = strong_scaling_main(img,ncores=8)
np.testing.assert_array_equal(res_single, res_thread)

time_threaded = %timeit -o strong_scaling_main(img,ncores=8)
print(f"Speedup: {time_ref_single.average / time_threaded.average}")

# %% [markdown]
"""
# Weak scaling

Now let’s try with a 1000x1000 image and 8 cores on weak scaling. For
the sake of the application, we simply replicate the image 8 times, and
apply the mean pooling to each image, one on each core.
"""

# %%
import functools
from multiprocessing.pool import ThreadPool
 
def meanpoolimg(img):
    . . .

def weak_scaling_main(imgs,ncores):
    . . .

N=1000
img=np.arange(N*N).reshape(N,N)

time_ref_single = . . .
time_threaded = %timeit -o weak_scaling_main([img]*8,ncores=8)
print(f"Speedup: {. . .}")

# %% [markdown]
"""
# The laws

## Getting the speedups
"""

# %%
import time
import pandas as pd

def weak_scaling_run(compute_func,ncores_range,N,verbose=True):
    cluster_times = []
    speedups = []
    img=np.arange(N*N).reshape(N,N)
    for nbr_parallel_blocks in ncores_range:
        imgs = [img]*nbr_parallel_blocks
        t1 = time.time()
        compute_func(imgs,nbr_parallel_blocks)
        total_time = time.time() - t1
        t2 = time.time()
        compute_func(imgs,1)
        total_time_seq = time.time() - t2
        if verbose:
            print(f"With {nbr_parallel_blocks} node(s): ")
            print("\tTime : {:.2f}s".format(total_time))
            print("\tTime seq : {:.2f}s".format(total_time_seq))
        cluster_times.append(total_time)
        speedups.append(total_time_seq/total_time)
    return pd.DataFrame({'time':cluster_times,'speedup':speedups},index=ncores_range)

def strong_scaling_run(compute_func,ncores_range,N,verbose=True):
    cluster_times = []
    speedups = []
    img=np.arange(N*N).reshape(N,N)
    for nbr_parallel_blocks in ncores_range:
        t1 = time.time()
        compute_func(img,nbr_parallel_blocks)
        total_time = time.time() - t1
        if verbose:
            print(f"With {nbr_parallel_blocks} node(s): ")
            print("\tTime : {:.2f}s".format(total_time))
        cluster_times.append(total_time)
        speedups.append(cluster_times[0]/total_time)
    return pd.DataFrame({'time':cluster_times,'speedup':speedups},index=ncores_range)


# %%
from sklearn.metrics import mean_squared_error
import plotly.express as px
# Function that takes as an argument the result of a run, a "law" function,
# its parameter s, which corresponds to the estimated "sequential" proportion of the run,
# the run description, the law description
# Displays the comparative view of the run with the law prediction
# displays the root-mean-square error between the two
# returns previous numerical data
def compare_law_and_run(run,law,ncores_range,s,run_description,law_description):
    law_pred = [law(N,s) for N in ncores_range]
    law_df = pd.DataFrame({law_description:law_pred},index=ncores_range)
    MSE = mean_squared_error(run['speedup'][1:],law_pred[1:])
    title = "{} vs {} s={:2f}".format(law_description,run_description,s)
    px.line(pd.concat([run.rename(columns={'speedup':run_description}),law_df],axis=1),
            y=[run_description,law_description],
            title=title,
            labels={"index":"Number of cores",
                    "value":"Speedup over 1 core",
                    "variable":"Law or Run"},
                width=600
        
        ).show()
    print("MSE :{:.2e}".format(MSE))
    return pd.Series({'run':run_description,'law':law_description,'s':s,'MSE':MSE}).to_frame().T


# %% [markdown]
r"""
## Amdahl’s law

Implement Amdahl’s law in a function `amdahl_func` that takes the number
of cores and the proportion of the code that can be parallelized as
arguments. The speedup is given by:

$$S_{up}=\frac{1}{s+\frac{1-s}{N}}$$
"""


# %%
# N = workers number
# S = "sequential part" of the code (proportion)
def amdahl_func(N,S):
    return ... # speedup 


# %% [markdown]
"""
## Make the scaling runs
"""

# %%
strong_run = strong_scaling_run(strong_scaling_main,range(1,9),10000)
weak_run = weak_scaling_run(weak_scaling_main,range(1,9),10000)

# %% [markdown]
"""
## Plot the results for strong scaling
"""

# %%
compare_law_and_run(strong_run,amdahl_func,range(1,9),0,"Strong run","Amdahl")

# %% [markdown]
"""
## Finding the real “S” part
"""


# %%
def find_s_amdahl(strong_run):
    return ... # s


# %% [markdown]
"""
## Now compare the law and the run with the real “s” part
"""

# %%
s_strong = find_s_amdahl(strong_run)
compare_law_and_run(strong_run,amdahl_func,range(1,9),s_strong,"Strong run","Amdahl")

# %% [markdown]
"""
## Apply it to weak scaling and gustafson’s law

Recalculate real s for weak scaling

## Discussion

Why the Sequential part is not the same for strong and weak scaling?
"""
