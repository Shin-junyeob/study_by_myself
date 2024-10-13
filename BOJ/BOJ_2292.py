import sys
n = int(sys.stdin.readline())
start = 1; num = 6; result = 1
if n == 1:
    print(1)
elif 1 < n <= 7:
    print(2)
else:
    while n > start:
        start += num
        num += 6
        result += 1
    print(result)