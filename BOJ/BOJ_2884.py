<<<<<<< HEAD
h, m = map(int, input().split())

if m >= 45:
    print(h, m-45)
else:
    if h == 0:
        h = 23
    else:
        h -= 1
=======
h, m = map(int, input().split())

if m >= 45:
    print(h, m-45)
else:
    if h == 0:
        h = 23
    else:
        h -= 1
>>>>>>> 77e016886 (Initial commit)
    print(h, m+15)