import sys

count = 1
while True:
    l, p, v = map(int, sys.stdin.readline().split())
    if (l == 0) and (p == 0) and (v == 0):
        break
    elif v % p < l:
        s = v // p
        print(f'Case {count}: {v-s*(p-l)}')
    else:
        num = p - (v % p)
        v += num
        s = v // p
        print(f'Case {count}: {v-s*(p-l)}')
    count += 1