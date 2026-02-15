import numpy as np
print('imported')
print(np.__version__)

# create 1-D array
arr1=np.array([23,45,67,89,44])
print(arr1)
print(type(arr1))

arr1=np.array([23,45,67,89,44,4.5,7.8])
print(arr1)
arr1=np.array([23,45,67,89,44,4.5,7.8,'abc'])
print(arr1)

# integer only
arr1=np.array([23,45,67,89,44,4.5,7.8],dtype='int')
print(arr1)

# create 2-d array
arr2=np.array([[1,2,3],
               [4,5,6]])

# check attributes
print('dim of array',arr2.ndim)
print('shape of array',arr2.shape)
print('data type of array',arr2.dtype)
print('itemsize of array',arr2.itemsize)
print('Number of elements of array',arr2.size)
print('No.of bytes of array',arr2.nbytes)

str_arr=np.array(['apple','banana','pineapple'])