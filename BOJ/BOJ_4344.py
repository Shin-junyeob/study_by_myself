import sys
c = int(sys.stdin.readline())
for i in range(c):
    lst = list(map(int, sys.stdin.readline().rstrip().split()))
    n = lst[0]
    lst = lst[1:]
    mean = sum(lst) / n
    count = 0
    for j in lst:
        if j > mean:
            count += 1
    result = (count / n) * 100
    print('{:.3f}'.format(result)+'%')