from __future__ import division
import pylab as plt
import numpy as np
import math
import leastsquares as ls

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

def verlet(h,y0,v):
    y0[0:2] = y0[:2] + h*v
    k = h*df(y0)[2:]
    y0[2:] = v+0.5*k
    v = v+k
    return np.array([y0,v])

def df(r):
    rad = np.sqrt(np.power(r[0],2)+np.power(r[1],2))
    df = np.zeros(4)
    df[0] = r[2]
    df[1] = r[3]
    df[2] = -4*np.power(np.pi, 2)*r[0]*np.power(rad,-3)
    df[3] = -4*np.power(np.pi, 2)*r[1]*np.power(rad,-3)
    return df

def tomin(x,T):
    a = 1.523679
    e = 0.0934
    semmaj = a/(1+e)
    x = x-e*np.sin(x)-T*np.sqrt(np.power(semmaj,-3))
    return x

def newraph(N,T):
    a = 1.523679
    e = 0.0934
    semmaj = a/(1+e)
    x = 0
    for i in range(N):
        x = x - (tomin(x,T))/(1-e*np.cos(x))
    r = semmaj*(1-e*np.cos(x))
    return r

def secant(N,T):
    a = 1.523679
    tol = 1e-15
    e = 0.0934
    semmaj = a/(1+e)
    x1 = 0
    x2 = np.pi*0.5
    for i in range(N):
        x = x2 - tomin(x2,T)*(x2-x1)/(tomin(x2,T)-tomin(x1,T))
        x1 = x2
        x2 = x
        if np.abs(x2-x1)<tol:
            break
    r = semmaj*(1-e*np.cos(x))
    return r

def relax(N,T):
    a = 1.523679
    e = 0.0934
    semmaj = a/(1+e)
    x = 0
    for i in range(N):
        x = x - tomin(x,T)
    r = semmaj*(1-e*np.cos(x))
    return r

def bisect(N,T):
    tol = 1e-16
    a = 1.523679
    e = 0.0934
    semmaj = a/(1+e)
    x1 = 0
    x2 = 2*np.pi
    fx1 = tomin(x1,T)
    fx2 = tomin(x2,T)
    count = 0
    while count<N:
        if np.sign(fx1).astype(int) == - np.sign(fx2).astype(int):
            xnew = 0.5*(x1+x2)
            fxnew = tomin(xnew,T)
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
    r = semmaj*(1-e*np.cos(x))
    return r

def orbital(N,T,i):
    h = T/N
    a = 1.523679
    e = 0.0934
    semmaj = a/(1+e)
    semmin = semmaj*np.sqrt(1-np.power(e,2))
    r = np.zeros(4)
    f = np.zeros(4)
    track = np.zeros([N,4])
    r[0] = -a*(1+e)
    r[3] = 2*np.pi*np.sqrt((1-e)/(a*(1+e)))
    if i == 0:
        for j in range(N):
            track[j,:] = r
            r = feul(h,r)
    if i == 1:
        for j in range(N):
            track[j,:] = r
            r = rk2(h,r)
    if i == 2:
        for j in range(N):
            track[j,:] = r
            r = rk4(h,r)
    if i == 3:
        track[0,:] = r
        vhalf = r[2:]+0.5*h*df(r)[2:]
        for j in range(N-1):
            inter = verlet(h,r,vhalf)
            r = inter[0]
            vhalf = inter[1]
            track[j+1,:] = r
    if i == 4:
        beta = 0.9
        ep0 = np.array([1e-12,1e-12,1e-12,1e-12])
        ep0min = min(ep0)
        t = 0
        track = [r]
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
                    track  = np.append(track,[r1],axis=0)
                    h = beta*h*np.power((ep0min/epmax),0.2)
                    r = r1
                    count = 0
                else:
                    h = beta*h*np.power((ep0min/epmax),0.25)
                    count += 1
    rad = np.sqrt(np.power(track[:,0],2)+np.power(track[:,1],2))
    E = .5*(np.power(track[:,3],2)+np.power(track[:,2],2))-np.power(rad,-1)
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(E,'k-')
    ax1[1].plot(track[:,0],track[:,1])
    return rad[-1]

def errorplot(N,T,filename=None):
    terms = len(N)
    a = np.zeros([terms,5])
    exact = newraph(1000,5)
    for i in range(terms):
        for j in range(5):
            a[i,j] = np.abs(orbital(N[i],T,j)-exact)
    lN = np.log10(N)
    la = np.log10(a)
    least  = np.zeros([5,2])
    for j in range(5):
        least[j,:] = ls.leastsquares(lN,la[:,j])
    
    fig1, ax1 = plt.subplots(1,1)
    # ax1.plot(lN, la[:,0],'ko')
    # p0, = ax1.plot(lN,least[0,0]*lN+least[0,1],'k-',label = '%.4f' % least[0,0])
    # ax1.set_title("t = "+str(T))
  
    # ax1.plot(lN, la[:,1],'ro')
    # p1, = ax1.plot(lN,least[1,0]*lN+least[1,1],'r-',label = '%.4f RK2' % least[1,0])

    # ax1.plot(lN, la[:,2],'bo')
    # p2, = ax1.plot(lN,least[2,0]*lN+least[2,1],'b-',label = '%.4f RK4' % least[2,0])

    # ax1.plot(lN, la[:,3],'go')
    # p3, = ax1.plot(lN,least[3,0]*lN+least[3,1],'g-',label = '%.4f Verlet' % least[3,0])

    # ax1.plot(lN, la[:,4],'mo')
    # p4, = ax1.plot(lN,least[4,0]*lN+least[4,1],'m-',label = '%.4f Adaptive RK4' % least[4,0])
    # l0 = ax1.legend(handles=[p1,p2,p3,p4],loc=1)  


    # ax1.set_ylabel(r'$r$ Log Error')
    # ax1[1].set_ylabel(r'RK2')
    # ax1[2].set_ylabel(r'RK4')
    # ax1[3].set_ylabel(r'Verlet')
    # ax1[4].set_ylabel(r'Adaptive RK4')
    # ax1[4].set_xlabel(r'Log $N$')
