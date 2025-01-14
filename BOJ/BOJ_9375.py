<<<<<<< HEAD
import sys
from collections import Counter

t = int(sys.stdin.readline())
for i in range(t):
    n = int(sys.stdin.readline().rstrip())
    lst = []
    for j in range(n):
        a, b = input().split()
        lst.append(b)
    answer = 1
    result = Counter(lst)
    for key in result:
        answer *= result[key] + 1
=======
import sys
from collections import Counter

t = int(sys.stdin.readline())
for i in range(t):
    n = int(sys.stdin.readline().rstrip())
    lst = []
    for j in range(n):
        a, b = input().split()
        lst.append(b)
    answer = 1
    result = Counter(lst)
    for key in result:
        answer *= result[key] + 1
>>>>>>> 77e016886 (Initial commit)
    print(answer - 1)