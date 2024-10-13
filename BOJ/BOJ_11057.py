import sys
n = int(sys.stdin.readline())
dp = [1 for _ in range(10)]

if n == 1:
    print(sum(dp))
if n >= 2:
    for i in range(n-1):
        for j in range(10):
            dp[j] += sum(dp[j:]) - dp[j]
    print(sum(dp)%10007)