import sys
x = int(sys.stdin.readline())
lst = [int(2**x) for x in range(6,-1,-1)]
while True:
    cnt = 0
    for i in range(len(lst)):
        if lst[i] <= x:
            x -= lst[i]
            cnt += 1
    if x == 0:
        print(cnt)
        exit()