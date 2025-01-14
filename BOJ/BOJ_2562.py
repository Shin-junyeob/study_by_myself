<<<<<<< HEAD
import sys

lst = [0 for _ in range(9)]
for i in range(9):
    lst[i] = int(sys.stdin.readline().rstrip())

print(max(lst))
=======
import sys

lst = [0 for _ in range(9)]
for i in range(9):
    lst[i] = int(sys.stdin.readline().rstrip())

print(max(lst))
>>>>>>> 77e016886 (Initial commit)
print(lst.index(max(lst))+1)