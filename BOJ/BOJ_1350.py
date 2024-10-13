import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]
m = int(sys.stdin.readline())

count = n
for i in range(n):
    if lst[i] == 0:
        count -= 1
    if lst[i] > m and lst[i] % m != 0:
        count += lst[i] // m
    if lst[i] > m and lst[i] % m == 0:
        count += lst[i] // m - 1

print(count * m)