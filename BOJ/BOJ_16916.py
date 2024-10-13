import sys, re
s = sys.stdin.readline().rstrip()
p = sys.stdin.readline().rstrip()
f = re.compile(p)
answer = f.findall(s)
if answer:
    print(1)
else:
    print(0)