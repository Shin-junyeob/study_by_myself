import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]

dp = [1]
for i in range(n-1):
    dp.append(0)
for i in range(1, n):
    arr = []
    for j in range(i):
       if lst[i] > lst[j]:
           arr.append(dp[j]+1)
    if arr:
        dp[i] += max(arr)
    else:
        dp[i] += 1

print(max(dp))