import numpy
import networks
sizes = [8, 6, 10]
print(sizes[1:])
print(sizes[:-1])
print()
# print([numpy.random.randn(3, 1)])
a = [numpy.random.randn(y, 1) for y in sizes[1:]]
b = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print(a)

# print(list(zip(sizes[:-1], sizes[1:])))

print(b)

print(list(zip(a, b)))

