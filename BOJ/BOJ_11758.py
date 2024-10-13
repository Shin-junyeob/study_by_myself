import sys
lst = [0 for _ in range(3)]
for i in range(3):
    lst[i] = list(map(int, sys.stdin.readline().rstrip().split()))

check = lst[0][0]*lst[1][1] + lst[1][0]*lst[2][1] + lst[2][0]*lst[0][1] - (lst[1][0]*lst[0][1] + lst[2][0]*lst[1][1] + lst[0][0]*lst[2][1])
if check > 0:
    print(1)
elif check < 0:
    print(-1)
else:
    print(0)