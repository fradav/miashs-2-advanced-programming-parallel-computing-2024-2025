# %% [markdown]
"""
# MultiProcessing, Strong Scaling

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Ghislain Durif (CNRS, LBMC)  
Monday, August 26, 2024

## ⚠️⚠️⚠️⚠️ Attention ⚠️⚠️⚠️⚠️

Under *Windows*, with python the multiprocessing module
`multiprocessing` works in a normal script but **not in notebooks**.

If you absolutely must use Windows,
use[WSL](https://docs.microsoft.com/fr-fr/windows/wsl/)

# Strong Scaling

## Prerequisites

For this TP, set the number of **physical** cores available (8 on the
cluster nodes), not the number of logical cores.
"""

# %%
ncores = 8 # 8 on the cluster nodes

# %%
import math

# %% [markdown]
r"""
# Introduction

## Basic functions

Make a function `is_prime` that tests whether an integer $n$ strictly
greater than 2 is prime or not.

Hint: First check that it is not even, then list all odd numbers from 3
to $\sqrt{n}$ (upper rounding) and test whether they are factors.

Make a function `total_primes` that counts the number of primes in a
list.

Calculate the number of primes from 1 to $N=100 000$ with this function
"""

# %%
N=100000

# %%
total_primes(range(1,N+1))

# %% [markdown]
"""
## Time measurement

Use `%timeit` to measure the average time taken to count the number of
primes up to $N=100000$. (note: by default, `timeit` repeats the
calculation $7times{}10$ to obtain a reliable average. Please refer to
the
[magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html)
and [timeit](https://docs.python.org/3.9/library/timeit.html) docs).

Store measurements using the -o option in timeit
"""

# %%
orig_time = %timeit -o total_primes(range(1,N+1))

# %% [markdown]
r"""
# First steps

Our first attempt at multiprocessing will involve partitioning the count
into 2. We’ll run two processes in parallel on $\{1,...,N/2\}$ and
$\{N/2+1,...,N\}$.

Complete the following code
([source](https://notebook.community/izapolsk/integration_tests/notebooks/MultiProcessing)).

Check the result and the performance gain with `%timeit`.
"""

# %%
from multiprocessing.pool import Pool

def split_total(N):
    with Pool(2) as pool:
        return sum(pool.map(total_primes, ...))


# %%
split_total(N)

# %%
split_time = %timeit -o split_total(N)

# %%
print("Gain with split : {:.1f}".format(orig_time.average/split_time.average))

# %% [markdown]
"""
# Generalization

Generalize the function with partitioning into *n* tasks instead of just
2. We’ll use a generalized `multi_process_list` function, which takes as
arguments : -
f`the main computation function, which takes an integer list as argument - n`
the number of partitions (here, one partition = task) -
par_f`a function which takes as argument a list and a number of partitions to be performed, and returns the list of partitions in this list - l`
the list to be partitioned
"""


# %%
def multi_process_list(f,n,par_f,l):
    with Pool(ncores) as pool:
        return sum(pool.map(...)


# %% [markdown]
"""
First, we write the `naive_par` partitioning function.
"""


# %%
def naive_par(lst,n):
    return ...


# %% [markdown]
"""
We’ll use the `chunks` function, which partitions a list into chunks of
fixed size (except for the last one).

We’ll test the gain obtained with 8 tasks/partitions.
"""


# %%
def chunks(lst, m):
    """Yield successive m-sized chunks from lst."""
    for i in range(0, len(lst), m):
        yield lst[i:i + m]


# %% [markdown]
"""
Vérifier le fonctionnement de `naive_par`
"""

# %%
multi_time = %timeit -o multi_process_list(total_primes,ncores,naive_par,range(1,N+1))

# %%
print("Gain avec multi : {:.1f}".format(orig_time.average/multi_time.average))

# %% [markdown]
"""
Repeat all calculations and payoff comparisons with $N=5000000$. To
avoid long calculation times, we’ll restrict ourselves to a single
iteration (option `-r 1 -n 1` in `%timeit`).
"""

# %%
N = 5000000

# %%
orig_time = %timeit -r 1 -n 1 -o total_primes(range(1,N+1))
split_time = %timeit -r 1 -n 1 -o split_total(N)
multi_time = %timeit -r 1 -n 1 -o multi_process_list(total_primes,ncores,naive_par,range(1,N+1))

print("Gain with split : {:.1f}".format(orig_time.average/split_time.average))
print("Gain with multi : {:.1f}".format(orig_time.average/multi_time.average))

# %% [markdown]
"""
# Optional refinement

How much time is spent on each task? Use the following function to get
an idea. What do you observe?
"""


# %%
def timed_total_primes(l):
    %timeit -r 1 -n 1 total_primes(l)
    return 0


# %% [markdown]
"""
How can we solve this problem?

Find a simple solution that requires only one line of code. Check the
execution times of individual tasks.

Compare again with $N = 10000000$ (which will take about 1 minute
sequentially).

# Recreational interlude

If you have a CPU with SMT (Hyperthreading), redo the measurements with
`ncores` equal to the number of logic cores, and explain the results.
"""
