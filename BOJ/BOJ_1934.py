import sys
t = int(sys.stdin.readline())
for _ in range(t):
    a, b = map(int, sys.stdin.readline().rstrip().split())
    ans = 0
    for i in range(1, min(a, b)+1):
        if a % i ==0 and b % i == 0:
            ans = max(ans, i)
    print(ans * (a//ans) * (b//ans))