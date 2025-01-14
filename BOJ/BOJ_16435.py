<<<<<<< HEAD
import sys
n, l = map(int, sys.stdin.readline().split())
lst = list(map(int, sys.stdin.readline().split()))
lst.sort()
for i in lst:
    if l >= i:
        l += 1

=======
import sys
n, l = map(int, sys.stdin.readline().split())
lst = list(map(int, sys.stdin.readline().split()))
lst.sort()
for i in lst:
    if l >= i:
        l += 1

>>>>>>> 77e016886 (Initial commit)
print(l)