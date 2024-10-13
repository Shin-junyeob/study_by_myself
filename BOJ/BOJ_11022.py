'''
메모리 제한 : 256MB = 256,000,000Byte
입력받는 수 t의 제한 : 1 <= t <= 64,000,000
'''

import sys
t = int(sys.stdin.readline())
if (1 <= t) and (t <= 64000000):
    for i in range(t):
        a, b = map(int, sys.stdin.readline().split())
        print(f'Case #{i+1}: {a} + {b} = {a+b}')