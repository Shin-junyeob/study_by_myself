import sys
from bisect import bisect_left, bisect_right

n = int(sys.stdin.readline())
lst = sorted(list(map(int, sys.stdin.readline().split())))
m = int(sys.stdin.readline())
check = list(map(int, sys.stdin.readline().split()))

def binary(lst, find):
    start = 0
    end = len(lst)-1
    while start <= end:
        pivot = (start+end)//2
        if lst[pivot] == find:
            return bisect_right(lst, find) - bisect_left(lst, find)
        elif lst[pivot] > find:
            end = pivot - 1
        else:
            start = pivot + 1
    return 0

for i in check:
    print(binary(lst, i), end = ' ')