<<<<<<< HEAD
import sys

a, b = map(int, sys.stdin.readline().split())
c = int(sys.stdin.readline())

if b + c >= 60:
    a = a + (b+c)//60
    if a >= 24:
        a = a - 24
    b = b+c - 60*((b+c)//60)
else:
    a = a
    b = b + c

=======
import sys

a, b = map(int, sys.stdin.readline().split())
c = int(sys.stdin.readline())

if b + c >= 60:
    a = a + (b+c)//60
    if a >= 24:
        a = a - 24
    b = b+c - 60*((b+c)//60)
else:
    a = a
    b = b + c

>>>>>>> 77e016886 (Initial commit)
print(a, b)