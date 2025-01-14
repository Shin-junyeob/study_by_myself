<<<<<<< HEAD
import sys
n = int(sys.stdin.readline())

def binary():
    start = 1
    end = n
    target = n
    while start <= end:
        pivot = (start+end)//2
        if pivot**2 == target:
            return pivot
        elif pivot**2 > target:
            end = pivot - 1
        else:
            start = pivot + 1

=======
import sys
n = int(sys.stdin.readline())

def binary():
    start = 1
    end = n
    target = n
    while start <= end:
        pivot = (start+end)//2
        if pivot**2 == target:
            return pivot
        elif pivot**2 > target:
            end = pivot - 1
        else:
            start = pivot + 1

>>>>>>> 77e016886 (Initial commit)
print(binary())