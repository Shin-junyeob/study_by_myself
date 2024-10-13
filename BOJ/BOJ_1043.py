import sys

n, m = map(int, sys.stdin.readline().split())
lst = [int(x) for x in sys.stdin.readline().split()]

know = set()
if 0 in lst:
    pass
else:
    know = know | set(lst[1:])

party = []
for i in range(m):
    arr = [int(x) for x in sys.stdin.readline().split()]
    party.append(arr[1:])

while True:
    length = len(know)
    for i in party:
        if set(i) & know:
            know = know | set(i)
    if len(know) == length:
        break

count = 0
for i in party:
    if not set(i) & know:
        count += 1

print(count)