import sys
code = list(map(int, sys.stdin.readline().rstrip()))

dp = [0 for _ in range(len(code)+1)]
dp[0] = dp[1] = 1
if code[0] == 0:
    print(0)
else:
    for i in range(2, len(code)+1):
        if code[i-1] > 0:
            dp[i] += dp[i-1]
        if 10 <= code[i-2]*10 + code[i-1] <= 26:
            dp[i] += dp[i-2]
    print(dp[-1] % 1000000)