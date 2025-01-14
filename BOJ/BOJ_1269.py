<<<<<<< HEAD
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

=======
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

>>>>>>> 77e016886 (Initial commit)
print(len(set(a)-set(b))+len(set(b)-set(a)))