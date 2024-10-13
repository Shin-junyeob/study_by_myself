import sys
n, l = map(int, sys.stdin.readline().split())
lst = list(map(int, sys.stdin.readline().split()))
lst.sort()
for i in lst:
    if l >= i:
        l += 1

print(l)