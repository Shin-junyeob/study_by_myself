<<<<<<< HEAD
import sys
string = sys.stdin.readline().rstrip()
dic = dict(zip(['apa','epe','ipi','opo','upu'], ['a','e','i','o','u']))
for i in dic:
    string = string.replace(i, dic[i])

=======
import sys
string = sys.stdin.readline().rstrip()
dic = dict(zip(['apa','epe','ipi','opo','upu'], ['a','e','i','o','u']))
for i in dic:
    string = string.replace(i, dic[i])

>>>>>>> 77e016886 (Initial commit)
print(string)