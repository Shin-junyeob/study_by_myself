<<<<<<< HEAD
import sys

lst = [0 for _ in range(10)]
for i in range(10):
    lst[i] = int(sys.stdin.readline().rstrip())

result = [-1 for _ in range(10)]
count = 0
for i in range(10):
    if lst[i] % 42 not in result:
        result[i] = lst[i] % 42
        count += 1

=======
import sys

lst = [0 for _ in range(10)]
for i in range(10):
    lst[i] = int(sys.stdin.readline().rstrip())

result = [-1 for _ in range(10)]
count = 0
for i in range(10):
    if lst[i] % 42 not in result:
        result[i] = lst[i] % 42
        count += 1

>>>>>>> 77e016886 (Initial commit)
print(count)