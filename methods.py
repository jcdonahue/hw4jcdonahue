from __future__ import division
import pylab as plt
import numpy as np
import math


def feul(h,y0):
    y = y0 + h*df(y0)
    return y

def rk2(h,y0):
    k1 = h*df(y0)
    k2 = h*df(y0+0.5*k1)
    y0 = y0 + k2
    return y0


def rk4(h,y0):
    k1 = h*df(y0)
    k2 = h*df(y0+0.5*k1)
    k3 = h*df(y0+0.5*k2)
    k4 = h*df(y0+k3)
    y0 = y0 + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
    return y0

def verlet(h,y1,y0):
    a = df(y1)
    y = np.zeros(4)
    y[0:2] = 2*y1[0:2] - y0[0:2] + a[2:]*np.power(h,2)
    return y

def df(r):
    rad = np.sqrt(np.power(r[0],2)+np.power(r[1],2))
    df = np.zeros(4)
    df[0] = r[2]
    df[1] = r[3]
    df[2] = -4*np.power(np.pi, 2)*r[0]*np.power(rad,-3)
    df[3] = -4*np.power(np.pi, 2)*r[1]*np.power(rad,-3)
    return df
        
def orbital(N,T,i):
    h = T/N
    a = 1.523679
    e = 0.0934
    r = np.zeros(4)
    f = np.zeros(4)
    track = np.zeros([N,2])
    r[0] = -a*(1+e)
    r[3] = 2*np.pi*np.sqrt((1-e)/(a*(1+e)))
    if i == 0:
        for j in range(N):
            track[j,:] = r[0:2]
            r = feul(h,r)
    if i == 1:
        for j in range(N):
            track[j,:] = r[0:2]
            r = rk2(h,r)
    if i == 2:
        for j in range(N):
            track[j,:] = r[0:2]
            r = rk4(h,r)
    if i == 3:
        r0 = r
        r1 = rk4(h,r)
        track[0,:] = r0[0:2]
        track[1,:] = r1[0:2]
        for j in range(N-2):
            r = verlet(h,r1,r0)
            track[j+2,:] = r[0:2]
            r0 = r1
            r1 = r
    if i == 4:
        beta = 0.9
        ep0 = np.array([1e-12,1e-12,1e-12,1e-12])
        ep0min = min(ep0)
        t = 0
        track = np.array([[r[0],0]])
        count = 0
        while t < T:
            if count > 100:
                print("I got stuck")
                break
            else:
                r1 = rk4(h,r)
                r2 = rk4(0.5*h,r)
                r3 = rk4(0.5*h,r2)
                ep = np.abs(r1-r3)/15
                epmax = max(ep)
                if np.all(ep0>ep):
                    t += h
                    track  = np.append(track,[r1[0:2]],axis=0)
                    h = beta*h*np.power((ep0min/epmax),0.2)
                    r = r1
                    count = 0
                else:
                    h = beta*h*np.power((ep0min/epmax),0.25)
                    count += 1
    plt.plot(track[:,0],track[:,1])
    return r
    
def newraph(t0,tf,y0,N,f,dy):
    h = abs(tf-t0)/N
    y = y0
    for i in range(N):
        y = y - f(y)/df(y)
    return y
