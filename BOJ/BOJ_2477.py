import sys
k = int(sys.stdin.readline())
lst = [0 for _ in range(6)]
max_w = 0; max_h = 0
for i in range(6):
    d, l = map(int, sys.stdin.readline().rstrip().split())
    lst[i] = l
    if d in [1, 2] and l > max_w:
        max_w = l
    if d in [3, 4] and l > max_h:
        max_h = l
max_w_idx = lst.index(max_w)
max_h_idx = lst.index(max_h)

ls = max_w * max_h
ss = abs(lst[(max_w_idx-1)%6] - lst[(max_w_idx+1)%6]) * abs(lst[(max_h_idx-1)%6] - lst[(max_h_idx+1)%6])
print((ls - ss) * k)