import numpy as np

a = np.array([[1, 1], [2, 3], [4, 5]])
b = np.array([1,1,1])

print(a.shape)
print(b.shape)
c=1
print("c: ",c)

d = np.size(b, 0)
print(d)

n=3
# print(range(n))
for i in range(n):
    print(i)
    print(a[i][1])

array_X = [[1,2], [3,4],[5,6],[7,8]]
vector = np.array([[1],[2],[3],[4],[5]])
print("vector's shape: ", vector.shape)
print("vector[0]: ", vector[0])
print(vector[:,0])

X=[[1,2],[3,4],[5,6]]
print("np.max: ", np.max(X))
print("np.min: ", np.min(X))
# print("np.linspace: ", np.linspace(np.min(X), np.max(X), 1000))
e = b.transpose()
print("e.shape: ", e.shape)
f, = e.shape
print(f)

g = [1,2,3,4,5,6,7,8,9]
print("g: ", g[1:4])