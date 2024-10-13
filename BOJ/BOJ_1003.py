import sys
n = int(sys.stdin.readline())

def fibonacci(n):
    dp = [0 for x in range(n + 1)]
    for i in range(n+1):
        if i == 0:
            dp[i] = [1, 0]
        elif i == 1:
            dp[i] = [0, 1]
        else:
            dp[i] = [dp[i-2][0] + dp[i-1][0], dp[i-2][1] + dp[i-1][1]]
    return dp[-1]

for i in range(n):
    s = int(sys.stdin.readline().rstrip())
    if (0 <= s) and (s <= 40):
        print(fibonacci(s)[0], fibonacci(s)[1])