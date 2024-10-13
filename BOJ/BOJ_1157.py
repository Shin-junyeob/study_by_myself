import sys, re
word = sys.stdin.readline().rstrip().upper()
p = re.compile('\w')
f = list(set(p.findall(word)))

max = 0
lst = []
for i in f:
    lst.append(word.count(i))
    if word.count(i) > max:
        max = word.count(i)
if lst.count(max) > 1:
    print('?')
else:
    print(f[lst.index(max)])