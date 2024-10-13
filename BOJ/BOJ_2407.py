import sys
n, m = map(int, sys.stdin.readline().split())

dp = [0 for x in range(m)]

for i in range(m):
    if i == 0:
        dp[i] = n
    else:
        dp[i] = dp[i-1] * (n-i) // (i+1)

print(dp[-1])