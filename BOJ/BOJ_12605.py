<<<<<<< HEAD
import sys
n = int(sys.stdin.readline())

for i in range(n):
    lst = sys.stdin.readline().rstrip().split(' ')
    ans = []
    while lst:
        word = lst.pop()
        ans.append(word)
=======
import sys
n = int(sys.stdin.readline())

for i in range(n):
    lst = sys.stdin.readline().rstrip().split(' ')
    ans = []
    while lst:
        word = lst.pop()
        ans.append(word)
>>>>>>> 77e016886 (Initial commit)
    print(f'Case #{i+1}: {" ".join(ans)}')