import sys, re
answer = re.fullmatch('(pi|ka|chu)*', sys.stdin.readline().rstrip())
print('YES' if answer != None else 'NO')