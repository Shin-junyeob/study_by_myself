<<<<<<< HEAD
n = int(input())
lst = [0 for x in range(n+1)]
for i in range(n+1):
    if i < 2:
        lst[i] = i
    else:
        lst[i] = lst[i-2] + lst[i-1]

=======
n = int(input())
lst = [0 for x in range(n+1)]
for i in range(n+1):
    if i < 2:
        lst[i] = i
    else:
        lst[i] = lst[i-2] + lst[i-1]

>>>>>>> 77e016886 (Initial commit)
print(lst[-1])