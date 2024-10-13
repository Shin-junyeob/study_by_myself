import sys
n, k = map(int, sys.stdin.readline().split())
dp = [0]*(n+1)
for i in range(1, k+1):
    for j in range(1, n+1):
        if j == 1:
            dp[j] = i
        else:
            dp[j] += dp[j-1]
print(dp[n]%1000000000)