import sys, re
word = sys.stdin.readline().rstrip()
s = re.split('(?=[A-Z])', word)
print(s)
count = 0
for i in range(1, len(s)-1):
    if len(s[i]) % 4 != 0:
        count += 4 - len(s[i]) % 4

print(count)