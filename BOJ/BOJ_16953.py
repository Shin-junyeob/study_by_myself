import sys
n, m = map(int, sys.stdin.readline().split())

count = 0
def func(n, m):
    global count
    while m > n:
        if m % 10 == 1:
            m = m // 10
            count += 1
        elif m % 2 == 0:
            m = m // 2
            count += 1
        else:
            return -1
    if n == m:
        return count+1
    else:
        return -1

print(func(n, m))