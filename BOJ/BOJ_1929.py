import sys
m, n = map(int, sys.stdin.readline().split())
check = [False, False] + [True]*(n-1)
primes = []
for i in range(2, n+1):
    if check[i]:
        if m <= i <= n:
            print(i)
        for j in range(2*i, n+1, i):
            check[j] = False