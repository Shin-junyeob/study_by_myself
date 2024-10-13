import sys, re
string = sys.stdin.readline().rstrip()
find = sys.stdin.readline().rstrip()
f = re.findall(find, string)
if f:
    print(len(f))
else:
    print(0)