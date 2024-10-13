import sys

n = int(sys.stdin.readline())
a = [int(x) for x in sys.stdin.readline().split()]
b = [int(x) for x in sys.stdin.readline().split()]

sum = 0
for i in range(n):
    sum += sorted(a, reverse = True)[i] * sorted(b, reverse = False)[i]

print(sum)