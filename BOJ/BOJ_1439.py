<<<<<<< HEAD
import sys

s = sys.stdin.readline().rstrip()
s_one = sorted(s.split('0'), reverse = True)
s_zero = sorted(s.split('1'), reverse = True)

for i in range(len(s_one)):
    if s_one[-1] == '':
        s_one.pop()

for i in range(len(s_zero)):
    if s_zero[-1] == '':
        s_zero.pop()

=======
import sys

s = sys.stdin.readline().rstrip()
s_one = sorted(s.split('0'), reverse = True)
s_zero = sorted(s.split('1'), reverse = True)

for i in range(len(s_one)):
    if s_one[-1] == '':
        s_one.pop()

for i in range(len(s_zero)):
    if s_zero[-1] == '':
        s_zero.pop()

>>>>>>> 77e016886 (Initial commit)
print(min(len(s_zero), len(s_one)))