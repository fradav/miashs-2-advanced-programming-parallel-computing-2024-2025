# %% [markdown]
"""
# Distributed models with dask

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Ghislain Durif (CNRS, LBMC)  
Monday, August 26, 2024

# Initialization
"""

# %%
from ipyparallel import Client

rc = Client()

# %% [markdown]
"""
`rc` is an interable of accessibles computing nodes.
"""

# %%
views = rc[:]

# %%
views

# %% [markdown]
"""
## Check cluster engines
"""

# %%
import platform
platform.node()

# %%
views.apply_sync(platform.node)

# %% [markdown]
"""
## Distributed prime numbers

Let’s revive our functions
"""

# %%
import math

def check_prime(n):
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


# %%
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# %%
def find_primes(r):
    return list(filter(check_prime,r))


# %% [markdown]
"""
## Peculiarities

You’ll have to -
[`push`](https://ipyparallel.readthedocs.io/en/latest/api/ipyparallel.html#ipyparallel.DirectView.push)
your dependant functions to the engines (`ipyparallel` does push your
main “mapped” function, but not its dependancies) - explicitly import
any required python library to the engines
"""

# %%
views.push({'check_prime': check_prime})

# %%
with views.sync_imports():
    import math

# %% [markdown]
"""
### First steps

1.  Complete with the correct
    [`views.map`](https://ipyparallel.readthedocs.io/en/latest/api/ipyparallel.html#ipyparallel.DirectView.map)
    call

``` python
def calculate_primes(N,chunksize):
    return dview.map_sync(check_prime , range(2,N),chunksize=chunksize)


1.  Benchmark it for

``` python
N = 5000000
chunksize = int(N/64)
```
"""

#%%
def calculate_primes(N,chunksize):
    return views.map_sync(find_primes ,chunks(range(1,N),chunksize))
# %%
N = 5000000

# %%
%timeit -r 1 -n 1 calculate_primes(N,int(N/64))

# %% [markdown]
"""
# (Aside) a network optimization : [broadcast_view](https://ipyparallel.readthedocs.io/en/latest/examples/broadcast/Broadcast%20view.html) (network optimization)
"""

# %% [raw]
"""
<center>
"""

# %% [markdown]
"""
<img src="attachment:image.png" width="500"/>
"""

# %% [raw]
"""
</center>
"""

# %%
direct_view = rc.direct_view()
bcast_view = rc.broadcast_view()

# %%
%timeit direct_view.apply_sync(lambda: None)

# %%
%timeit bcast_view.apply_sync(lambda: None)

# %% [markdown]
"""
# An embarrasingly parallel example : distributed Monte-Carlo computing of $\pi$

If we sample randomly a bunch of $N$ points in the unity square, and
counts all points $N_I$ verifying the condition

$x^2 + y^2 \le 1$ whichs means they are in the upper right quarter of a
disk.

We have this convergence

$\lim_{N\to\infty} 4\frac{N_I}{N} = \pi$
"""

# %% [raw]
"""
<center>
"""

# %% [markdown]
"""
<img src="attachment:hpp2_0901.png" width="40%" />
"""

# %% [raw]
"""
</center>
"""

# %% [markdown]
"""
### 2. Write the function which :

-   takes a number of estimates `nbr_estimates` as argument
-   samples them in the \[(0,0),(1,1)\] unity square
-   returns the number of points inside the disk quarter

``` python
def estimate_nbr_points_in_quarter_circle(nbr_estimates):
    ...
    return nbr_trials_in_quarter_unit_circle
```

### 3. Make it distributed

-   Wraps the previous function in
    `python     def calculate_pi_distributed(nnodes,nbr_samples_in_total)         ...         return estimated_pi`
-   `nnodes` will use only `rc[:nnodes]` and split the number of
    estimates for each worker nodes into `nnodes` blocks.
-   Try it on `1e8` samples and benchmark it on 1 to 8 nodes. (use
    [`time`](https://docs.python.org/3/library/time.html#time.time))
-   Plot the performance gain over one node and comment the plot.

$\Longrightarrow$ We see a near perfect linear scalability.
"""

#%%
import random
with views.sync_imports():
    import random
def estimate_nbr_points_in_quarter_circle(nbr_estimates):
    nbr_trials_in_quarter_unit_circle = 0
    for _ in range (int(nbr_estimates)):
        x, y = random.uniform(0,1), random.uniform(0,1)
        if x**2 + y**2 <= 1 :
            nbr_trials_in_quarter_unit_circle += 1
    return nbr_trials_in_quarter_unit_circle
# %%
4*estimate_nbr_points_in_quarter_circle(1e4)/1e4
# %%
def calculate_pi_distributed(nnodes,nbr_samples_in_total):
    res = rc[:nnodes].map_sync(estimate_nbr_points_in_quarter_circle,[int(nbr_samples_in_total/nnodes)]*nnodes)
    n1 = sum(res)
    estimated_pi = 4*(n1/nbr_samples_in_total)
    return estimated_pi
# %%
calculate_pi_distributed(8,1e7)
# %%
import time

N = 1e8
cluster_times = []
pis = []
for nbr_parallel_blocks in range(1,9):
    print(f"With {nbr_parallel_blocks} node(s): ")
    t1 = time.time()
    pi_estimate = calculate_pi_distributed(nbr_parallel_blocks,N)
    total_time = time.time() - t1
    print(f"\tPi estimate : {pi_estimate}")
    print("\tTime : {:.2f}s".format(total_time))
    cluster_times.append(total_time)
    pis.append(pi_estimate)
# %%
import plotly.express as px

speedups_cores = [cluster_times[0]/cluster_times[i] for i in range(8)]
px.line(y=speedups_cores,x=range(1,9),
        labels={"x":"Number of cores",
                "y":"Speedup over 1 core"},
       width=600)
# %%
