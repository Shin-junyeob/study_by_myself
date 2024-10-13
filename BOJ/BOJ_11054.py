import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]
dp = [1 for _ in range(n)]
dpr = [1 for _ in range(n)]
for i in range(1, n):
    for j in range(i):
        if lst[j] < lst[i]:
            dp[i] = max(dp[i], dp[j]+1)

lst.reverse()
for i in range(1, n):
    for j in range(i):
        if lst[j] < lst[i]:
            dpr[i] = max(dpr[i], dpr[j]+1)

dpr.reverse()
max = 0
for i in range(n):
    if dp[i] + dpr[i] > max:
        max = dp[i] + dpr[i]

print(max-1)