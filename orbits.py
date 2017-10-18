from __future__ import division
import pylab as plt
import numpy as np
import math
import time
import leastsquares as ls
from methods import feul,rk2,rk4,verlet,df,tomin,secant,newraph,relax,bisect,adaprk4


def orbital(N,T,i):
    h = T/N
    a = 1.523679
    e = 0.0934
    r = np.zeros(4)
    r[0] = -a*(1+e)
    r[3] = 2*np.pi*np.sqrt((1-e)/(a*(1+e)))
    if i == 0:
        track = feul(h,r,N)
    if i == 1:
        track = rk2(h,r,N)
    if i == 2:
        track = rk4(h,r,N)
    if i == 3:
        track = verlet(h,r,N)
    if i == 4:
        track = adaprk4(h,r,T)
    rad = np.sqrt(np.power(track[:,0],2)+np.power(track[:,1],2))
    E = .5*(np.power(track[:,3],2)+np.power(track[:,2],2))-4*np.pi*np.power(rad,-1)
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(E,'k-')
    ax1[1].plot(track[:,0],track[:,1])
    return rad[-1]

def halley(N,pers):
    period = 75
    T = pers*period
    h = T/N
    a = 17.834144
    e = .96714
    r = np.zeros(4)
    r[0] = -a*(1+e)
    r[3] = 2*np.pi*np.sqrt((1-e)/(a*(1+e)))
    t1 = time.time()
    track1 = rk4(h,r,N)
    t2 = time.time()
    track2 = verlet(h,r,N)
    t3 = time.time()
    track3 = adaprk4(h,r,T)
    t4 = time.time()
    #rad = np.sqrt(np.power(track[:,0],2)+np.power(track[:,1],2))
    #E = .5*(np.power(track[:,3],2)+np.power(track[:,2],2))-np.power(rad,-1)
    fig1, ax1 = plt.subplots(1,1)
    #ax1[0].plot(E,'k-')
    ax1.plot(track1[:,0],track1[:,1],'b-')
    ax1.plot(track2[:,0],track2[:,1],'g-')
    ax1.plot(track3[:,0],track3[:,1],'k-')
    return (t2-t1,t3-t2,t4-t3)


def errorplot(N,T,filename=None):
    terms = len(N)
    a = np.zeros([terms,5])
    exact = newraph(1000,5,1.523679,0.0934)
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
