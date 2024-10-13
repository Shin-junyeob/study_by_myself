import sys
n = sys.stdin.readline().rstrip()

lst = [int(x) for x in n]
lst.sort(reverse = True)

if (sum(lst) % 3 == 0) and (0 in lst):
    for i in range(len(lst)):
        print(lst[i], end = '')
else:
    print(-1)