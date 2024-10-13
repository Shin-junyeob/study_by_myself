import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]

print(min(lst), max(lst))