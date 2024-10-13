import sys
n, m = map(int, sys.stdin.readline().split())
lst = [int(x) for x in sys.stdin.readline().split()]

dp = [0 for _ in range(n+1)]
for i in range(n+1):
    if i == 0:
        dp[i] = 0
    else:
        dp[i] = dp[i-1] + lst[i-1]

for _ in range(m):
    i, j = map(int, sys.stdin.readline().split())
    print(dp[j] - dp[i-1])