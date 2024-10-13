import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]

div = [int(x) for x in range(max(lst), -max(lst)-1, -1)]

def func():
    ans = [0 for _ in range(n)]
    for i in div:
        if i != 0:
            for j in range(len(lst)):
                ans[j] = lst[j] % i
            if len(set(ans)) == 1:
                print(i)
                exit()

print(func())