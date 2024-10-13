import sys
xs, ys = map(int, sys.stdin.readline().split())
xe, ye, dx, dy = map(int, sys.stdin.readline().split())

def dis():
    s = 0
    dis = (xs-xe)**2 + (ys-ye)**2
    while True:
        if dx != 0:
            d = dy / dx
            s += 1
            nx = xe + s
            ny = ye + s*d
            if dis > (xs-nx)**2 + (ys-ny)**2:
                dis = (xs-nx)**2 + (ys-ny)**2
            else:
                print(xe+s-1, int(ye+(s-1)*d))
                exit()
        else:
            s += 1
            nx = xe
            ny = ye + s
            if dis > (xs-nx)**2 + (ys-ny)**2:
                dis = (xs-nx)**2 + (ys-ny)**2
            else:
                print(xe, int(ye+s-1))
                exit()
dis()