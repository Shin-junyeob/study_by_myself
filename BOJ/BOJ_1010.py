import sys
t = int(sys.stdin.readline())
for i in range(t):
    n, m = map(int, sys.stdin.readline().split())
    dp = [0 for x in range(n+1)]
    if n != m:
        for i in range(n+1):
            if i == 0 :
                dp[i] = 0
            elif i == 1:
                dp[i] = m//i
            else:
                dp[i] = dp[i-1] * (m-i+1)//i
        ans = dp[-1]
    else:
        ans = 1

    print(ans)