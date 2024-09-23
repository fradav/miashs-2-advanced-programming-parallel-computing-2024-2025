# %% [markdown]
r"""
# Dask `delayed` App

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Ghislain Durif (CNRS, LBMC)  
Monday, August 26, 2024

<img src="https://docs.dask.org/en/latest/_images/dask_horizontal.svg"
     align="right"
     width="30%"
     alt="Dask logo\">

# dask.delayed - parallelize any code

What if you don’t have an array or dataframe? Instead of having blocks
where the function is applied to each block, you can decorate functions
with `@delayed` and *have the functions themselves be lazy*.

This is a simple way to use `dask` to parallelize existing codebases or
build [complex
systems](https://blog.dask.org/2018/02/09/credit-models-with-dask).

**Related Documentation**

-   [Delayed
    documentation](https://docs.dask.org/en/latest/delayed.html)
-   [Delayed screencast](https://www.youtube.com/watch?v=SHqFmynRxVU)
-   [Delayed API](https://docs.dask.org/en/latest/delayed-api.html)
-   [Delayed examples](https://examples.dask.org/delayed.html)
-   [Delayed best
    practices](https://docs.dask.org/en/latest/delayed-best-practices.html)

As we’ll see in the [distributed scheduler
notebook](05_distributed.ipynb), Dask has several ways of executing code
in parallel. We’ll use the distributed scheduler by creating a
`dask.distributed.Client`. For now, this will provide us with some nice
diagnostics. We’ll talk about schedulers in depth later.

### Iniatilization of the ipyparallel/dask interface
"""

# %%
from ipyparallel import Client

### Force the 8 nodes/1 core layout
rc = Client()
rc.stop_distributed()
client = rc.become_dask()
client

# %% [markdown]
"""
## A Typical Workflow

Typically if a workflow contains a for-loop it can benefit from delayed.
The following example outlines a read-transform-write:

``` python
import dask
    
@dask.delayed
def process_file(filename):
    data = read_a_file(filename)
    data = do_a_transformation(data)
    destination = f"results/{filename}"
    write_out_data(data, destination)
    return destination

results = []
for filename in filenames:
    results.append(process_file(filename))
    
dask.compute(results)
```

## Basics

First let’s make some toy functions, `inc` and `add`, that sleep for a
while to simulate work. We’ll then time running these functions
normally.

In the next section we’ll parallelize this code.
"""

# %%
from time import sleep


def inc(x):
    sleep(1)
    return x + 1


def add(x, y):
    sleep(1)
    return x + y


# %% [markdown]
"""
We time the execution of this normal code using the `%%time` magic,
which is a special function of the Jupyter Notebook.
"""

# %%
%%time
# This takes three seconds to run because we call each
# function sequentially, one after the other

x = inc(1)
y = inc(2)
z = add(x, y) 

# %% [markdown]
"""
### Parallelize with the `dask.delayed` decorator

Those two increment calls *could* be called in parallel, because they
are totally independent of one-another.

We’ll make the `inc` and `add` functions lazy using the `dask.delayed`
decorator. When we call the delayed version by passing the arguments,
exactly as before, the original function isn’t actually called yet -
which is why the cell execution finishes very quickly. Instead, a
*delayed object* is made, which keeps track of the function to call and
the arguments to pass to it.
"""

# %%
import dask


@dask.delayed
def inc(x):
    sleep(1)
    return x + 1


@dask.delayed
def add(x, y):
    sleep(1)
    return x + y


# %%
%%time
# This runs immediately, all it does is build a graph

x = inc(1)
y = inc(2)
z = add(x, y)

# %% [markdown]
"""
This ran immediately, since nothing has really happened yet.

To get the result, call `compute`. Notice that this runs faster than the
original code.
"""

# %%
%%time
# This actually runs our computation using a local thread pool

z.compute()

# %% [markdown]
"""
## What just happened?

The `z` object is a lazy `Delayed` object. This object holds everything
we need to compute the final result, including references to all of the
functions that are required and their inputs and relationship to
one-another. We can evaluate the result with `.compute()` as above or we
can visualize the task graph for this value with `.visualize()`.
"""

# %%
z

# %%
# Look at the task graph for `z`
z.visualize()

# %% [markdown]
"""
Notice that this includes the names of the functions from before, and
the logical flow of the outputs of the `inc` functions to the inputs of
`add`.

### Some questions to consider:

-   Why did we go from 3s to 2s? Why weren’t we able to parallelize down
    to 1s?
-   What would have happened if the inc and add functions didn’t include
    the `sleep(1)`? Would Dask still be able to speed up this code?
-   What if we have multiple outputs or also want to get access to x or
    y?

## Exercise: Parallelize a for loop

`for` loops are one of the most common things that we want to
parallelize. Use `dask.delayed` on `inc` and `sum` to parallelize the
computation below:
"""

# %%
data = [1, 2, 3, 4, 5, 6, 7, 8] 

# %%
%%time
# Sequential code


def inc(x):
    sleep(1)
    return x + 1


results = []
for x in data:
    y = inc(x)
    results.append(y)

total = sum(results)

# %%
total

# %%
%%time
# Your parallel code here...

# %% [markdown]
"""
How do the graph visualizations compare with the given solution,
compared to a version with the `sum` function used directly rather than
wrapped with `delayed`? Can you explain the latter version? You might
find the result of the following expression illuminating

``` python
inc(1) + inc(2)
```

## Putting it all together with a the producer-consumer pattern

## Introduction with a simple producer-consumer example

We’ll try to reproduce the producer/consumer code from asynchronous
application with `asyncio` or `threading` but with `dask.delayed` and
compute

``` python
from dask import delayed, compute
from dask.distributed import Queue, print
from time import sleep

queue = Queue()
...
```

## Exercise: Parallelize a for-loop code with control flow

Often we want to delay only *some* functions, running a few of them
immediately. This is especially helpful when those functions are fast
and help us to determine what other slower functions we should call.
This decision, to delay or not to delay, is usually where we need to be
thoughtful when using `dask.delayed`.

In the example below we iterate through a list of inputs. If that input
is even then we want to call `inc`. If the input is odd then we want to
call `double`. This `is_even` decision to call `inc` or `double` has to
be made immediately (not lazily) in order for our graph-building Python
code to proceed.
"""


# %%
def double(x):
    sleep(1)
    return 2 * x

def inc(x):
    sleep(1)
    return x + 1

def is_even(x):
    return not x % 2


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# %%
%%time
# Sequential code

results = []
for x in data:
    if is_even(x):
        y = double(x)
    else:
        y = inc(x)
    results.append(y)

total = sum(results)
print(total)

# %% [markdown]
"""
``` python
%%time
# Your parallel code here...
# TODO: parallelize the sequential code above using dask.delayed
# You will need to delay some functions, but not all
```

### Some questions to consider:

-   What are other examples of control flow where we can’t use delayed?

-   What would have happened if we had delayed the evaluation of
    `is_even(x)` in the example above?

-   What are your thoughts on delaying `sum`? This function is both
    computational but also fast to run.

-   Example of control flow with we can’t use delayed : conditional
    loops, recursive functions

-   Nothing, we have to compute it immediately for branching

-   Delaying sum isn’t worth the graph walk overload

## Exercise: Parallelize a Pandas Groupby Reduction

In this exercise we read several CSV files and perform a groupby
operation in parallel. We are given sequential code to do this and
parallelize it with `dask.delayed`.

The computation we will parallelize is to compute the mean departure
delay per airport from some historical flight data. We will do this by
using `dask.delayed` together with `pandas`. In a future section we will
do this same exercise with `dask.dataframe`.

## Create data

Run this code to prep some data.

This downloads and extracts some historical flight data for flights out
of NYC between 1990 and 2000. The data is originally from
[here](http://stat-computing.org/dataexpo/2009/the-data.html).
"""

# %%
import os
import requests

os.makedirs("data", exist_ok=True)
code_url = "https://raw.githubusercontent.com/dask/dask-tutorial/main/prep.py"
with open("prep.py", "wb") as f:
    f.write(requests.get(code_url).content)
%run prep.py -d flights

# %% [markdown]
"""
### Inspect data
"""

# %%
sorted(os.listdir(os.path.join("data", "nycflights")))

# %% [markdown]
"""
### Read one file with `pandas.read_csv` and compute mean departure delay
"""

# %%
import pandas as pd

df = pd.read_csv(os.path.join("data", "nycflights", "1990.csv"))
df.head()

# %%
# What is the schema?
df.dtypes

# %%
# What originating airports are in the data?
df.Origin.unique()

# %%
# Mean departure delay per-airport for one year
df.groupby("Origin").DepDelay.mean()

# %% [markdown]
"""
### Sequential code: Mean Departure Delay Per Airport

The above cell computes the mean departure delay per-airport for one
year. Here we expand that to all years using a sequential for loop.
"""

# %%
from glob import glob

filenames = sorted(glob(os.path.join("data", "nycflights", "*.csv")))

# %%
filenames

# %%
%%time

sums = []
counts = []
for fn in filenames:
    # Read in file
    df = pd.read_csv(fn)

    # Groupby origin airport
    by_origin = df.groupby("Origin")

    # Sum of all departure delays by origin
    total = by_origin.DepDelay.sum()

    # Number of flights by origin
    count = by_origin.DepDelay.count()

    # Save the intermediates
    sums.append(total)
    counts.append(count)

# Combine intermediates to get total mean-delay-per-origin
total_delays = sum(sums)
n_flights = sum(counts)
mean = total_delays / n_flights

# %%
mean

# %% [markdown]
"""
### Parallelize the code above

Use `dask.delayed` to parallelize the code above. Some extra things you
will need to know.

1.  Methods and attribute access on delayed objects work automatically,
    so if you have a delayed object you can perform normal arithmetic,
    slicing, and method calls on it and it will produce the correct
    delayed calls.

2.  Calling the `.compute()` method works well when you have a single
    output. When you have multiple outputs you might want to use the
    `dask.compute` function. This way Dask can share the intermediate
    values.

So your goal is to parallelize the code above (which has been copied
below) using `dask.delayed`. You may also want to visualize a bit of the
computation to see if you’re doing it correctly.

``` python
%%time
# your code here
```
"""

# %%
# ensure the results still match
mean

# %% [markdown]
"""
### Some questions to consider:

-   How much speedup did you get? Is this how much speedup you’d expect?
-   Experiment with where to call `compute`. What happens when you call
    it on `sums` and `counts`? What happens if you wait and call it on
    `mean`?
-   Experiment with delaying the call to `sum`. What does the graph look
    like if `sum` is delayed? What does the graph look like if it isn’t?
-   Can you think of any reason why you’d want to do the reduction one
    way over the other?

### Learn More

Visit the [Delayed
documentation](https://docs.dask.org/en/latest/delayed.html). In
particular, this [delayed
screencast](https://www.youtube.com/watch?v=SHqFmynRxVU) will reinforce
the concepts you learned here and the [delayed best
practices](https://docs.dask.org/en/latest/delayed-best-practices.html)
document collects advice on using `dask.delayed` well.

Some improvements by actual parallelization of file reading, but we’re
limited by IO throughput.

<span class="proof-title">*Solution*. </span>Delaying the sum could be
interesting in case to make more balanced loads between tasks.

## Close the Client

Before moving on to the next exercise, make sure to close your client or
stop this kernel.
"""

# %%
client.close()
