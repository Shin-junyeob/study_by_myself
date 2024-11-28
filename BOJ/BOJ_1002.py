import sys
t = int(sys.stdin.readline())
lst = [0 for _ in range(t)]
for i in range(t):
    lst[i] = list(map(int, sys.stdin.readline().rstrip().split()))

def func(lst):
    x1, y1, r1, x2, y2, r2 = lst.pop(0)
    if (x1, y1) == (x2, y2):
        if r1 == r2:
            print(-1)
        else:
            print(0)
    else:
        dis = ((x2 - x1)**2 + (y2 - y1)**2)**(0.5)
        dis2 = r2 + r1
        dis3 = abs(r2 - r1)
        if dis == dis2 or dis == dis3:
            print(1)
        elif dis < dis2 and dis > dis3:
            print(2)
        else:
            print(0)

for i in range(t):
    func(lst)
    