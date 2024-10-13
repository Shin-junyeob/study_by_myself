import sys
n = int(sys.stdin.readline())
lst = [[0, 0] for _ in range(n)]
for i in range(1, n+1):
    t, p = map(int, sys.stdin.readline().rstrip().split())
    if t + i <= n+1:
        lst[i-1] = [t, p]

dp = [0 for _ in range(n)]
lstr = lst[::-1]
for i in range(n):
    if i == 0:
        dp[i] = lstr[i][1]
    else:
        dp[i] += max(dp[i-(lstr[i][0])] + lstr[i][1], dp[i-1])

print(dp[-1])