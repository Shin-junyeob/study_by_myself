import sys
n = int(sys.stdin.readline())
arr = []
dp = []
for i in range(n):
    num = int(sys.stdin.readline().rstrip())
    arr.append(num)
    dp.append(1)

for i in range(1, n):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j]+1)

print(n - max(dp))