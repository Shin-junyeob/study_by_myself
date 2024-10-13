price = int(input())
count = int(input())
for i in range(count):
    a, b = map(int, input().split())
    price -= a * b
print('Yes' if price == 0 else 'No')