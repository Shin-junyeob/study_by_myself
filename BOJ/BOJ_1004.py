import sys
t = int(sys.stdin.readline())
for i in range(t):
    count = 0
    x1, y1, x2, y2 = map(int, sys.stdin.readline().rstrip().split())
    n = int(sys.stdin.readline())
    for j in range(n):
        x, y, r = map(int, sys.stdin.readline().rstrip().split())
        ds1 = ((x-x1)**2 + (y-y1)**2)**(0.5)
        ds2 = ((x-x2)**2 + (y-y2)**2)**(0.5)
        if ds1 <= r and ds2 <= r:
            pass
        elif ds1 <= r and ds2 > r:
            count += 1
        elif ds1 > r and ds2 <= r:
            count += 1
        else:
            pass
    print(count)