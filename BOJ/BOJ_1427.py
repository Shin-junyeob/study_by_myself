n = input()
lst = [0 for x in range(len(n))]

for i in range(len(n)):
    lst[i] = n[i]

lst.sort(reverse = True)
ans = ""
for i in range(len(n)):
    ans = ans +lst[i]

print(int(ans))