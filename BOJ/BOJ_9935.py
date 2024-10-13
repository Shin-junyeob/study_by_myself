import sys, re
string = sys.stdin.readline().rstrip()
find = sys.stdin.readline().rstrip()
stack = []

for i in range(len(string)):
    stack.append(string[i])
    if stack[-1] == find[-1] and len(stack) >= len(find):
        if stack[-len(find):] == list(find):
            for j in range(len(find)):
                stack.pop()

if stack:
    print(''.join(stack))
else:
    print('FRULA')