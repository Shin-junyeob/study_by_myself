import sys
n = int(sys.stdin.readline())
lst = [int(x) for x in sys.stdin.readline().split()]
lst.sort()

s, e = 0, n-1
mmin, mmax = lst[s], lst[e]
answer = abs(mmin + mmax)

while s < e:
    ans = lst[s] + lst[e]
    if abs(ans) < answer:
        answer = abs(ans)
        mmin = lst[s]; mmax = lst[e]
        if answer == 0:
            break
    elif ans < 0:
        s += 1
    else:
        e -= 1

print(mmin, mmax)