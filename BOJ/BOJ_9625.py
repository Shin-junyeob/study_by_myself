import sys
k = int(sys.stdin.readline())
dp = [0 for _ in range(k+1)]
for i in range(k+1):
    if i == 0:
        dp[i] = [1, 0]
    if i >= 1:
        dp[i] = [dp[i-1][1], dp[i-1][0] + dp[i-1][1]]

print(dp[-1][0], dp[-1][1])