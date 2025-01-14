<<<<<<< HEAD
import sys, re
answer = re.fullmatch('(pi|ka|chu)*', sys.stdin.readline().rstrip())
=======
import sys, re
answer = re.fullmatch('(pi|ka|chu)*', sys.stdin.readline().rstrip())
>>>>>>> 77e016886 (Initial commit)
print('YES' if answer != None else 'NO')