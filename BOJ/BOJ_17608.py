import sys
n = int(sys.stdin.readline())
stack = [0 for x in range(n)]
for i in range(n):
    stack[i] = int(sys.stdin.readline().rstrip())

ans = []
def func(lst):
    wall = lst.pop()
    ans.append(wall)
    while lst:
        wall = lst.pop()
        if wall > ans[-1]:
            ans.append(wall)
        if wall <= ans[-1]:
            continue
    return len(ans)

print(func(stack))