import sys
n = int(sys.stdin.readline())
lst = sorted(list(map(int, sys.stdin.readline().split())))
m = int(sys.stdin.readline())
check = list(map(int, sys.stdin.readline().split()))

def binary(lst, target):
    start = 0
    end = len(lst)-1
    while start <= end:
        pivot = (start+end)//2
        if lst[pivot] == target:
            return 1
        elif lst[pivot] > target:
            end = pivot-1
        else:
            start = pivot+1
    return 0

for i in check:
    print(binary(lst, i), end = ' ')