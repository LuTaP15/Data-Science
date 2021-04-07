import numpy as np

# The Basics

# Create an array
a = np.array([1, 2, 3])

b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])


# Get Dimension
print(a.ndim)

# Get Shape
print(a.shape)

# Get Type --> 'int32' is default
print(a.dtype)

# Get Size
print(a.itemsize)

# Get total size
print(a.size * a.itemsize)
print(a.nbytes)

####################################################################################
# Accessing/Changing specific elements
a = np.array([[1,2,3,4,5,6,7], [8,9,10,11,12,13,14]])

# Get a specific element [row, column]
print(a[1, 5])

# Get a specific row
print(a[0, :])

# Get a specific column
print(a[:, 2])

# Getting specific values with a stepsize [startindex:endindex:stepsize]
print(a[0, 1:6:2])

# Change a specific element
a[1, 5] = 20
print(a)

a[:, 2] = [98, 99]
print(a)

# 3D example
b = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
print(b)

# Get a specific element (work outside in)
print(b[0, 1, 1])
print(b[:, 1, :])

# Replace in 3D
b[:,1,:] = [[9,9], [8,8]]
print(b)

####################################################################################

# Initializing different types of arrays

# all zero matrix
print(np.zeros((2,3)))

# all ones matrix
print(np.ones((4,2,2)))

# Any other number
print(np.full((2,2), 99))

# Any other number (full_like)
print(np.full_like(a, 44))

# Random decimal numbers
print(np.random.rand(4,2))
print(np.random.random_sample((a.shape)))

# Random Integer values
print(np.random.randint(7, size=(3,3)))
print(np.random.randint(-2, 8, size=(3,3)))

# Identity Matrix
print(np.identity(5))

# Repeat an array
arr = np.array([[1,2,3]])
r1 = np.repeat(arr, 3, axis=0)
print(r1)
r2 = np.repeat(arr, 3, axis=1)
print(r2)

####################################################################################

# Be careful when copying arrays

# Wrong way
a = np.array([1,2,3])
b = a
b[0] = 100
print(a)
print(b)

# Right way
a = np.array([1,2,3])
b = a.copy()
b[0] = 100
print(a)
print(b)

# https://www.youtube.com/watch?v=GB9ByFAIAH4
# 36 min