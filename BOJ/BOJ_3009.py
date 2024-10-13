import sys
lst = [0 for _ in range(3)]
x_list = []
y_list = []
for i in range(3):
    x1, y1 = map(int, sys.stdin.readline().rstrip().split())
    lst[i] = [x1, y1]
    x_list.append(x1)
    y_list.append(y1)

for i in range(3):
    if x_list.count(x_list[i]) == 1:
        target_x = x_list[i]
    if y_list.count(y_list[i]) == 1:
        target_y = y_list[i]

print(target_x, target_y)