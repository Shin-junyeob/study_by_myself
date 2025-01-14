<<<<<<< HEAD
import sys

n = int(sys.stdin.readline())
a = [int(x) for x in sys.stdin.readline().split()]
b = [int(x) for x in sys.stdin.readline().split()]

sum = 0
for i in range(n):
    sum += sorted(a, reverse = True)[i] * sorted(b, reverse = False)[i]

=======
import sys

n = int(sys.stdin.readline())
a = [int(x) for x in sys.stdin.readline().split()]
b = [int(x) for x in sys.stdin.readline().split()]

sum = 0
for i in range(n):
    sum += sorted(a, reverse = True)[i] * sorted(b, reverse = False)[i]

>>>>>>> 77e016886 (Initial commit)
print(sum)