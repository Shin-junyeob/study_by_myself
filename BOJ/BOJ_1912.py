import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]

dp = [0 for _ in range(n)]
dp[0] = lst[0]
for i in range(1, n):
    if lst[i] + dp[i-1] > lst[i]:
        dp[i] += lst[i] + dp[i-1]
    else:
        dp[i] += lst[i]

print(max(dp))