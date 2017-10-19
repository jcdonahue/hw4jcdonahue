from __future__ import division
import pylab as plt
import numpy as np
import math
import leastsquares as ls

def feul(h,y0,N):
    track = np.zeros([N,4])
    track[0,:] = y0
    for j in range(N-1):
        y0 = y0 + h*df(y0)
        track[j+1,:] = y0
    return track

def rk2(h,y0,N):
    track = np.zeros([N,4])
    track[0,:] = y0
    for j in range(N-1):
        k1 = h*df(y0)
        k2 = h*df(y0+0.5*k1)
        y0 = y0 + k2
        track[j+1,:] = y0
    return track

def rk4(h,y0,N):
    track = np.zeros([N,4])
    track[0,:] = y0
    for j in range(N-1):
        k1 = h*df(y0)
        k2 = h*df(y0+0.5*k1)
        k3 = h*df(y0+0.5*k2)
        k4 = h*df(y0+k3)
        y0 = y0 + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
        track[j+1,:] = y0
    return track

def adaprk4(h,r,T):
    ep0 = np.array([1e-10,1e-10,1e-10,1e-10])
    t = 0
    track = [r]
    count = 0
    while t < T:
        if count > 100:
            print("I got stuck")
            break
        else:
            r1 = rk4(2*h,r,2)[-1]
            r2 = rk4(h,r,3)[-1]
            diff = np.abs(r1-r2)
            if np.any(diff == 0):
                diff = h*ep0
            rho = 30*h*ep0/diff
            rhomax = max(rho)
            if np.all(rho>1):
                t += h
                track  = np.append(track,[r1],axis=0)
                p = np.power(min(rho),0.25)
                if p>2:
                    p=2
                h = h*p
                r = r1
                count = 0
            else:
                p = np.power(min(rho),0.25)
                h = h*p
                count += 1
    return track

def verlet(h,r,N):
    track = np.zeros([N,4])
    track[0,:] = r
    vhalf = r[2:]+0.5*h*df(r)[2:]
    y0 = np.zeros(4)
    for j in range(N-1):
        y0[0:2] = r[:2] + h*vhalf
        k = h*df(r)[2:]
        y0[2:] = vhalf+0.5*k
        vhalf = vhalf+k
        r = y0
        track[j+1,:] = r
    return track

def df(r):
    rad = np.sqrt(np.power(r[0],2)+np.power(r[1],2))
    df = np.zeros(4)
    df[0] = r[2]
    df[1] = r[3]
    df[2] = -4*np.power(np.pi, 2)*r[0]*np.power(rad,-3)
    df[3] = -4*np.power(np.pi, 2)*r[1]*np.power(rad,-3)
    return df

def tomin(x,T,a,e):
    semmaj = a/(1+e)
    x = x-e*np.sin(x)-T*np.sqrt(np.power(semmaj,-3))
    return x

def newraph(N,T,a,e):
    # a = 1.523679
    # e = 0.0934
    x = 0
    for i in range(N):
        x = x - (tomin(x,T,a,e))/(1-e*np.cos(x))
    r = a*(1-e*np.cos(x))
    return r

def secant(N,T):
    a = 1.523679
    tol = 1e-15
    e = 0.0934
    x1 = 0
    x2 = np.pi*0.5
    for i in range(N):
        x = x2 - tomin(x2,T,a,e)*(x2-x1)/(tomin(x2,T,a,e)-tomin(x1,T,a,e))
        x1 = x2
        x2 = x
        if np.abs(x2-x1)<tol:
            break
    r = a*(1-e*np.cos(x))
    return r

def relax(N,T):
    a = 1.523679
    e = 0.0934
    x = 0
    for i in range(N):
        x = x - tomin(x,T,a,e)
    r = a*(1-e*np.cos(x))
    return r

def bisect(N,T):
    tol = 1e-16
    a = 1.523679
    e = 0.0934
    x1 = 0
    x2 = 2*np.pi
    fx1 = tomin(x1,T,a,e)
    fx2 = tomin(x2,T,a,e)
    count = 0
    while count<N:
        if np.sign(fx1).astype(int) == - np.sign(fx2).astype(int):
            xnew = 0.5*(x1+x2)
            fxnew = tomin(xnew,T,a,e)
            if  np.sign(fx1).astype(int) == np.sign(fxnew).astype(int):
                x1 = xnew
                fx1 = fxnew
                count += 1
                if abs(x1-x2)<tol:
                    break
            else:
                x2 = xnew
                fx2 = fxnew
                count += 1
                if abs(x1-x2)<tol:
                    break
        else:
            print('even root number at %i!' % count)
            break
    x = 0.5*(x1+x2)
    r = a*(1-e*np.cos(x))
    return r
