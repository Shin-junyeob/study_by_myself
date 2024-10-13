import sys
word = sys.stdin.readline().rstrip()

cnt = 0
length = 0
for i in ['c=', 'c-', 'd-', 'lj', 'nj', 's=', 'z=']:
    if i in word:
        cnt += word.count(i)
length = cnt
if 'dz=' in word:
    length += word.count('dz=')
print(len(word) - length)