<<<<<<< HEAD
import sys, re
string = sys.stdin.readline().rstrip()
find = sys.stdin.readline().rstrip()
f = re.findall(find, string)
if f:
    print(len(f))
else:
=======
import sys, re
string = sys.stdin.readline().rstrip()
find = sys.stdin.readline().rstrip()
f = re.findall(find, string)
if f:
    print(len(f))
else:
>>>>>>> 77e016886 (Initial commit)
    print(0)