

a = '92.38 96.08 86.93 88.21 95.44'
b = a.split(' ')
print(b)
b = list(map(float, b))
print(b)
print(sum(b)/5)