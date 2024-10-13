import sys
n = int(sys.stdin.readline())

stack = []
for i in range(n):
    inp = sys.stdin.readline().rstrip().split()
    if len(stack) == 0:
        if 'top' in inp:
            print(-1)
        elif 'pop' in inp:
            print(-1)
        elif 'size' in inp:
            print(0)
        elif 'empty' in inp:
            print(1)
        elif 'push' in inp:
            stack.append(int(inp[-1]))
    else:
        top = stack[-1]
        size = len(stack)
        if 'top' in inp:
            print(top)
        elif 'pop' in inp:
            stack.pop()
            print(top)
        elif 'size' in inp:
            print(size)
        elif 'empty' in inp:
            print(0)
        elif 'push' in inp:
            stack.append(int(inp[-1]))