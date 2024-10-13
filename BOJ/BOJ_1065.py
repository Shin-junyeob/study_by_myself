import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in range(1,n+1)]

def func(n):
    count = 0
    if n < 100:
        count += n
    if 100 <= n <= 1000:
        for i in range(100, n+1):
            if int(str(i)[2]) - int(str(i)[1]) == int(str(i)[1]) - int(str(i)[0]):
                count += 1
        count += 99
    return count

print(func(n))