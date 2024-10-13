import sys
t = int(sys.stdin.readline())
for i in range(t):
    lst = []
    n = int(sys.stdin.readline().rstrip())
    for j in range(n):
        a, b = map(int, sys.stdin.readline().rstrip().split())
        lst.append([a, b])
    lst.sort()
    cnt = 1
    max = lst[0][1]
    for i in range(1, n):
        if lst[i][1] < max:
            cnt += 1
            max = lst[i][1]
    print(cnt)