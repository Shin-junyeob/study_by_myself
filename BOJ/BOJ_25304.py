<<<<<<< HEAD
price = int(input())
count = int(input())
for i in range(count):
    a, b = map(int, input().split())
    price -= a * b
=======
price = int(input())
count = int(input())
for i in range(count):
    a, b = map(int, input().split())
    price -= a * b
>>>>>>> 77e016886 (Initial commit)
print('Yes' if price == 0 else 'No')