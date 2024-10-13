lst = []
for i in range(9):
    lst.append(int(input()))

over = (sum(lst)) - 100

ind = []
for i in range(9):
    for j in range(9):
        if (i != j) & (lst[i] + lst[j] == over):
            ind.append(i)
            ind.append(j)

del lst[ind[1]]
del lst[ind[0]]

lst.sort()

for i in range(len(lst)):
    print(lst[i])