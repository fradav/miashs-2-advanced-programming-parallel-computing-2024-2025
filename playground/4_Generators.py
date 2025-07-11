# %% [markdown]
"""
# A tutorial on Python generators

François-David Collin (CNRS, IMAG, Paul-Valéry Montpellier 3
University)  
Ghislain Durif (CNRS, LBMC)  
Monday, August 26, 2024

## Generators

A generator is essentially an iterator over an object (say a dataset).
You get a small chunk of data obtained through “iterating over the
larger object” every time you make a call to the generator. Generators
might prove to be useful in your implementation of sequential training
algorithms where you only require a few samples of your data. For
example, in a mini batch stochastic gradient descent, you would need to
generate random samples from the dataset for performing an update on
your gradient. Generators can be used in such use cases to create memory
efficient implementations of your algorithm, since they allow you to
perform operations without loading the whole dataset.

Also see PEP 255 (https://www.python.org/dev/peps/pep-0255/). The
explanation presented here is quite thorough.

### Behaviour of generators

A generator behaves like a function with states. Typically, functions in
Python do not have any state information. The variables defined within
the function scope are reset/destroyed at the end of every function
call. A generator allows you store intermediate states between calls, so
that every subsequent call can resume from the last state of execution.
Generators introduced the `yield` keyword to Python. We will look at a
few examples below.

**NOTE**

Although generators use the `def` keyword, they are not function
objects. Generators are a class in their own right, but are slightly
different from function objects.

We take a look at our first generator.
"""


# %%
## Example from PEP 0255
def fib():
    a, b = 0, 1
    while 1:
        yield b
        a, b = b, a + b


# %% [markdown]
"""
This is a generator that yields the infinite Fibonnaci sequence. With
every call to fib after the first call, the state of the generator gets
updated and the value of `b` is returned.

To use a generator, we first create an instance of the generator. Use
the `next` keywork to make calls to the generator. Once a generator has
been consumed completely, a `StopIteration` is raised if you try to
consume more elements from the generator.
"""

# %%
gen1 = fib()

# prints the first 10 fibonnaci numbers
for i in range(10):
    print(next(gen1), end=', ')
print("\nPassed!")

# %% [markdown]
"""
This example shows how you can represent an infinte sequence in Python
without using up all the memory in the world. Next, we will look at a
more practical example.
"""


# %%
def nsquared(n):
    while True:
        yield n ** 2
        n = n - 1
        if n == 0:
            return  # correct way to terminate a generator

#%%
sum(nsquared(10))
# %%
gen2 = nsquared(10)

for i in gen2:
    print(i, end=', ')

try:
    next(gen2) # should raise a StopIteration exception
except StopIteration:
    print("\nWe hit the the end of the generator, no more elements can be consumed")
except Exception as e:
    print("\nOops! Unexpected error", e)
finally:
    print("Passed !")

# %% [markdown]
"""
Now, suppose you want to find the sum of squares of the first 1,000,000
(1 million) integers. You don’t believe the analytical formula and want
to calculate it directly by summing up all the requisite squares of
integers. It is not memory efficient to create a list of 1 million
integers just to compute a sum. This is where our custom generator comes
to our rescue.
"""

# %%
squared_sum1 = sum([i**2 for i in range(1000001)])
print(squared_sum1)

# %%
gen3 = nsquared(1000000)
squared_sum2 = sum(gen3)
print(squared_sum2)

assert squared_sum1 == squared_sum1, "Sums are not equal !"
print("Passed !")

# %% [markdown]
"""
Although both snippets of code give the same result, the implementation
with the generator is more scalable since it uses constant memory.

### Generator expressions

See PEP 289 (https://www.python.org/dev/peps/pep-0289/).

Generator expressions merge the concepts of both generators and list
comprehensions. The syntax is almost similar to list comprehensions but
the returned result is a generator instead of a list.
"""

# %%
gen4 = nsquared(10)
print(gen4)
gen5 = (i**2 for i in range(11))
print(gen5)

# %% [markdown]
"""
Both generators and generator expressions can be passed to the tuple,
set or list constructors to create equivalent tuples, sets or lists.

**NOTE** - I strongly recommend using finite generators in such use
cases.
"""

# %%
# note that the generator has to be reinitialized once it has been consumed
gen4 = nsquared(10)
print(tuple(gen4))
gen4 = nsquared(10)
print(list(gen4))
gen4 = nsquared(10)
print(set(gen4))

print(tuple(i**2 for i in range(11)))
print(list(i**2 for i in range(11)))
print(set(i**2 for i in range(11)))

# %% [markdown]
"""
All the rules discussed in the previous sections about conditionals also
apply to generator expressions
"""

# %%
import numpy as np
print(list(i**2 for i in range(11) if i <=5))
print(list(i**2 if i <=5 else 1 for i in range(11)))
mat = list(i**2 + j**2 if i < j else i + j for i in range(3) for j in range(3))
print(np.array(mat).reshape(3,3))

# %% [markdown]
"""
### Advanced generator stuff

See PEP 380 for details. (https://www.python.org/dev/peps/pep-0380/)

Python 3 introduced the concept of one generator delegating to
sub-generators. This is achieved with the use of the `yield from`
keyword.

Suppose, you want to create a fancy new sequence by concatenating 3
sequences - the Fibonnaci sequence, a geometric series and a constant
series. You can do this by creating a generator that delegates each of
the subsequences to their own generators. To do this, we first create
our subsequence generators.
"""


# %%
# Same function, redefined here for clarity
def fib(n):
    a, b = 0, 1
    count = 0
    while 1:
        yield b
        count += 1
        if count == n:
            return
        a, b = b, a + b

def geom(n):
    a = 1
    count = 0
    while True:
        yield a
        count += 1
        if count == n:
            return
        a = a * 2

def constant(n):
    count = 0
    while True:
        yield -1
        count += 1
        if count == n:
            return


# %% [markdown]
"""
Now, we define our master generator.
"""


# %%
def master_sequence(n):
    g1 = fib(n)
    g2 = geom(n)
    g3 = constant(n)
    count = 0
    
    yield from g1
    yield from g2
    yield from g3


# %%
master_gen = master_sequence(5) # creates a sequence of length 15
print(list(master_gen))

# %% [markdown]
"""
#### A non-trivial example

Here is a non-trivial example of generator used in the Keras API
(https://keras.io/preprocessing/image/). The flow_from_directory method
returns a generator that yields batches of image data indefinitely. This
generator delegates the process to subgenerators that in turn yield data
from subfolders created in your dataset. Using this generator, you can
analyze very large image datasets on your PC without loading the entire
dataset into your RAM. This data generator is used to feed neural nets
during training using variations of gradient descent.
"""
