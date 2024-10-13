import sys

lst = [0 for _ in range(9)]
for i in range(9):
    lst[i] = int(sys.stdin.readline().rstrip())

print(max(lst))
print(lst.index(max(lst))+1)