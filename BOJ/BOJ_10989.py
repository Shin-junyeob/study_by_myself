import sys
n = int(sys.stdin.readline())
check = [0 for _ in range(10001)]

for _ in range(n):
    check[int(sys.stdin.readline())] += 1

for i, x in enumerate(check):
    if x != 0:
        for j in range(x):
            print(i)