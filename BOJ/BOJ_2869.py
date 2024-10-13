import sys
a, b, v = map(int, sys.stdin.readline().split())
d = a - b; day = (v-a)/d
if day % 1 != 0:
    print(int(day)+2)
else:
    print(int(day)+1)