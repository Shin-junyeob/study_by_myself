import sys
t = int(sys.stdin.readline())
for _ in range(t):
    x, y = map(int, sys.stdin.readline().split())
    dis = y-x
    cnt = 0
    move = 1
    length = 0
    while length < dis:
        cnt += 1
        length += move
        if cnt % 2 == 0:
            move += 1
    print(cnt)