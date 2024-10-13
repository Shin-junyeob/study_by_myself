import sys
n = int(sys.stdin.readline())
lst = sorted(list(map(int, sys.stdin.readline().split())))
m = int(sys.stdin.readline())
check = list(map(int, sys.stdin.readline().split()))

def binary(find):
    start = 0
    end = n-1
    while start <= end:
        p = (start+end)//2
        if lst[p] == find:
            return 1
        elif lst[p] > find:
            end = p-1
        else:
            start = p+1
    return 0

for i in check:
    print(binary(i))