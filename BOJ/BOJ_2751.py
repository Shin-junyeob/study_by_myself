<<<<<<< HEAD
import sys
n = int(sys.stdin.readline().rstrip())
lst = []
for i in range(n):
    lst.append(int(sys.stdin.readline().rstrip()))

lst.sort()
for i in range(n):
=======
import sys
n = int(sys.stdin.readline().rstrip())
lst = []
for i in range(n):
    lst.append(int(sys.stdin.readline().rstrip()))

lst.sort()
for i in range(n):
>>>>>>> 77e016886 (Initial commit)
    print(lst[i])