import sys
s = int(sys.stdin.readline())

n = int((2*s)**0.5 - 1)
lst = [1 for x in range(n+1)]
for i in range(n+1):
    lst[i] += i

if sum(lst) < s:
    lst[-1] += s - sum(lst)
elif sum(lst) > s:
    lst.pop()
    lst[-1] += s - sum(lst)

print(len(lst))