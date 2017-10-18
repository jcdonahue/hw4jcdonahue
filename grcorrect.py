from __future__ import division
import pylab as plt
import numpy as np
import math
import time
import leastsquares as ls


def dfgr(r,i):
    rad = np.sqrt(np.power(r[0],2)+np.power(r[1],2))
    rs = i*2.0/63200
    df = np.zeros(4)
    df[0] = r[2]
    df[1] = r[3]
    df[2] = -4*np.power(np.pi, 2)*r[0]*np.power(rad-rs,-2)/rad
    df[3] = -4*np.power(np.pi, 2)*r[1]*np.power(rad-rs,-2)/rad
    return df

def rk4gr(h,y0,N,i):
    track = np.zeros([N,4])
    track[0,:] = y0
    for j in range(N-1):
        k1 = h*dfgr(y0,i)
        k2 = h*dfgr(y0+0.5*k1,i)
        k3 = h*dfgr(y0+0.5*k2,i)
        k4 = h*dfgr(y0+k3,i)
        y0 = y0 + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
        track[j+1,:] = y0
    return track


def adaprk4gr(h,r,T,i):
    beta = 0.9
    ep0 = np.array([1e-10,1e-10,1e-10,1e-10])
    ep0min = min(ep0)
    t = 0
    track = [r]
    count = 0
    while t < T:
        if count > 100:
            print("I got stuck")
            break
        else:
            r1 = rk4gr(h,r,2,i)[-1]
            r2 = rk4gr(0.5*h,r,3,i)[-1]
            ep = np.abs(r1-r2)/15
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
    return track


def orbitalgr(N,pers):
    period = 15.559
    T = period*pers
    h = T/N
    e = 0.880
    a = 120
    rs = 2.0/63200
    r = np.zeros(4)
    r[0] = -a*(1-e)
    r[3] = 2*np.pi*np.sqrt((1-e)/(a*(1+e)))
                                    
    track1 = adaprk4gr(h,r,T,0)
    track2 = adaprk4gr(h,r,T,1)
                                    
    rad1 = np.sqrt(np.power(track1[:,0],2)+np.power(track1[:,1],2))
    rad2 = np.sqrt(np.power(track2[:,0],2)+np.power(track2[:,1],2))
    E1 = .5*(np.power(track1[:,3],2)+np.power(track1[:,2],2))-np.power(rad1,-1)
    E2 = .5*(np.power(track2[:,3],2)+np.power(track2[:,2],2))-np.power(rad2-rs,-1)
    fig1, ax1 = plt.subplots(2,1)
    ax1[0].plot(E1,'k-')
    ax1[0].plot(E2,'r-')
    ax1[1].plot(track1[:,0],track1[:,1],'k-')
    ax1[1].plot(track2[:,0],track2[:,1],'r-')
    return np.abs(rad1[-1]-rad2[-1])
