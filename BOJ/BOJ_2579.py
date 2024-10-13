import sys
n = int(sys.stdin.readline())
lst = [0 for x in range(n)]
for i in range(n):
    lst[i] = int(sys.stdin.readline().rstrip())

dp = [0 for x in range(n)]

for i in range(n):
    if i == 0:
        dp[i] = lst[i]
    elif i == 1:
        dp[i] = dp[i-1] + lst[i]
    elif i == 2:
        dp[i] = max(lst[i-1] + lst[i], lst[i-2] + lst[i])
    else:
        dp[i] = max(dp[i-2] + lst[i], dp[i-3] + lst[i-1] + lst[i])

print(dp[-1])