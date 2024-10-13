import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]
lst.sort()

if len(lst) % 2 == 0:
    print(lst[0] * lst[-1])
else:
    print(lst[len(lst)//2]**2)