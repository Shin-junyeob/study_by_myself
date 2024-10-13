import sys, re
from collections import deque

t = int(sys.stdin.readline())
for _ in range(t):
    p = sys.stdin.readline().rstrip()
    n = int(sys.stdin.readline())
    f = deque(re.findall('\w+', sys.stdin.readline().rstrip()))

    r = 0
    try:
        for i in p:
            if i == 'R':
                r += 1
            else:
                if r % 2 == 0:
                    f.popleft()
                else:
                    f.pop()
        if r % 2 == 0:
            print('['+','.join(f)+']')
        else:
            f.reverse()
            print('['+','.join(f)+']')
    except:
        print('error')