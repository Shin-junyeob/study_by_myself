import sys, re
s = sys.stdin.readline().rstrip()
lst = [-1 for _ in range(26)]
for i in range(len(s)):
    if lst[ord(s[i])-97] == -1:
        lst[ord(s[i])-97] = i

for i in lst:
    print(i, end=' ')