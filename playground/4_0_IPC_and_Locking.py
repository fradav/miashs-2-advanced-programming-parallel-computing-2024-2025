# %% [markdown]
"""
# IPC and Locking

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Ghislain Durif (CNRS, LBMC)  
Monday, August 26, 2024

# Streaming (pipelining) data

Sticking to our favorite hobby, which is finding prime numbers, this
time, we’ll use a different strategy.

Instead of partitioning the data from scratch, we will continuously
*feed* workers from our multiprocessing pool with small size chunk of
numbers and the workers send us back the the primes they found on those
chunks.

We need two queues : one for the chunks of numbers that’s the *INPUT*

Another one for the results sent that’s the *OUTPUT*
"""

# %% [markdown]
"""
<pre style="font-size:0.4em">
                                                              ┌──────────────┐
                                                              │              │
                                                              │              │
                           ┌─────────────────────────────────►│   Worker 1   ├───────────────────────────────────┐
                           │                                  │              │                                   │
                           │                                  │              │                                   │
                           │                                  └──────────────┘                                   │
                           │                                                                                     │
                           │                                  ┌──────────────┐                                   │
                           │                                  │              │                                   │
                           │                                  │              │                                   │
                           │                    ┌────────────►│   Worker 2   ├───────────────────────────────────┼────────────────────┐
                           │                    │             │              │                                   │                    │
                           │                    │             │              │                                   │                    │
              ┌──────┐  ┌──┴───┐  ┌──────┐  ┌───┴──┐          └──────────────┘                      ┌──────┐  ┌──▼───┐  ┌──────┐  ┌───▼──┐
...           │Chunk4│  │Chunk3│  │Chunk2│  │Chunk1│                                 ...            │Res. 4│  │Res. 3│  │Res. 2│  │Res. 1│
              └───┬──┘  └──────┘  └───┬──┘  └──────┘          ┌──────────────┐                      └───▲──┘  └──────┘  └───▲──┘  └──────┘
                  │                   │                       │              │                          │                   │
                  │                   │                       │              │                          │                   │
                  │                   └──────────────────────►│   Worker 3   ├──────────────────────────┼───────────────────┘
                  │                                           │              │                          │
                  │                                           │              │                          │
                  │                                           └──────────────┘                          │
                  │                                                                                     │
                  │                                           ┌──────────────┐                          │
                  │                                           │              │                          │
                  │                                           │              │                          │
                  └──────────────────────────────────────────►│   Worker 4   ├──────────────────────────┘
                                                              │              │
                                                              │              │
       ──────────────────────────────────────────►            └──────────────┘               ──────────────────────────────────────────►

                      INPUT                                                                               OUTPUT
</pre>
"""

# %% [markdown]
"""
Let’s revive our old `check_prime` function back from the dead…
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


# %% [markdown]
"""
Let’s get back the chunk generator, too.
"""


# %%
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# %% [markdown]
"""
Now we want to use a *producer/consumer* model, where each process got :

-   an *input* queue for receiving chunks of numbers
-   an *output* queue for sending back a list of all primes found in the
    *input* queue.

Both *input* and *output* elements are *lists* (or even python iterables
for the input). We’ll use `None` as a terminating element in the queue.

## Queue Worker Function

Create a *worker* function, which takes as the *input* and *output*
queues as argument.

For each element in the *input* queue, which is a list of numbers, get
the primes (as a list). Put the list of found primes to the *output*
queue.

``` python
def find_prime_worker(input, output):
    for chunk in iter(input.get,None):
        primes_found = ...
        output.put(primes_found)
```

## Test the worker function

1.  Manually allocate the *input* and *output* queues (we use managed
    queues)
2.  Put some chunks of numbers in the *input* queue (don’t forget to
    terminate the queue with `None`)
3.  Launch the worker function on the queues and terminate the output
    queue with `None`.
4.  Collect the results in a unified list.

## Some Tools

### Iterate on a queue

To make a queue terminated by `None` iterable use the
[`iter`](https://docs.python.org/3/library/functions.html#iter) function
:

``` python
iter(queue.get,None)
```

### Collect a list of list

To collect a list of list use the
[`chain`](https://docs.python.org/3/library/itertools.html#itertools.chain)
function from `itertools` :

``` python
chain(*list_of_list)
```

### Reminder

Iterables are lazy in python, to actually make a list you have to force
a `list()` of them.

``` python
list(iterables)
```

### Worker function

### 1. Allocations

### 2. Some chunk in the input

### 3. Launch the worker and terminate the output

### 4. Collect the results

# Putting the workers to… work.

make a function which allocates the queues, and use a `Pool(ncore)` of
worker.

``` python
def calculate_primes(ncore,N,chunksize):
    ...
```

-   `ncore` is the number of workers (and will be aligned with the
    number of cores you got, 8 for example)
-   `N` is the upper limit of the primes we want to find
-   `chunksize` is the size of the chunks we’ll send to process to
    workers.

## The main process

1.  First we’ll use a
    [`starmap_async`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap_async)
    for our main dispatcher function (don’t use the `chunksize` optional
    argument of the function)
2.  Feed the input queue with all chunks from the
    `chunks(range(1,N),chunksize)`
3.  Terminate the *input* queue (`ncores * None`, one for each worker)
4.  Wait for the workers to finish
5.  Collect and return the results

Test and benchmark it on a `int(N/64)` chunk size

``` python
N = 5000000
```
"""

#%%
def find_prime_worker(input, output):
    for chunk in iter(input.get,None):
        primes_found = list(filter(check_prime, chunk))
        output.put(primes_found)

#%%
from multiprocessing import Manager 
manager = Manager()
input = manager.Queue()
output = manager.Queue()


#%%
input.put(range(2,100))
input.put(range(1000,2000))
input.put(None)

#%%
find_prime_worker(input, output)
output.put(None)
#%%
from itertools import chain
list(chain(* list(iter(output.get, None))))
#%%
## Solution for main process function

### Test of the main function
"""

# %%
N = 5000000

# %% [markdown]
"""
## Autosizing chunks ! (optional)

We know that greater the number is, longer it is to check if it is a
prime.

A slight optimization to our multi-processing/queued algorithm is to
make the chunks smaller and smaller with greater numbers chunks.

1.  Try to modify the `chunks` function to take this into account, test
    the function.
2.  Modify the `calculate_primes_chunks` to use this function
3.  Test and benchmark it.

### autosizing chunks example function

### Test it

### Modify the main worker process function

### Test and benchmark it
"""
