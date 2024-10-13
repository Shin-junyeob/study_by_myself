import sys
t = int(sys.stdin.readline())
for i in range(t):
    v, e = list(map(int, sys.stdin.readline().split()))
    n = e - v + 2
    print(n)