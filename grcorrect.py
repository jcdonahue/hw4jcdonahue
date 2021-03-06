from __future__ import division
import pylab as plt
import numpy as np
import math
import time
import leastsquares as ls


def dfgr(r,i):            #the same as df in the regular code with the option to add in GR effects
    rad = np.sqrt(np.power(r[0],2)+np.power(r[1],2))
    rs = i*2.0/63200
    df = np.zeros(4)
    df[0] = r[2]
    df[1] = r[3]
    df[2] = -4*np.power(np.pi, 2)*r[0]*np.power(rad-rs,-2)/rad
    df[3] = -4*np.power(np.pi, 2)*r[1]*np.power(rad-rs,-2)/rad
    return df

def rk4gr(h,y0,N,i):        #same as previous rk4 but an option to add in GR effects
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



def adaprk4gr(h,r,T,i):      #same as other adaptive scheme but with GR effects optional
    ep0 = np.array([1e-14,1e-14,1e-14,1e-14])
    t = 0
    track = [r]
    count = 0
    while t < T:
        if count > 100:
            print("I got stuck")
            break
        else:
            r1 = rk4gr(2*h,r,2,i)[-1]
            r2 = rk4gr(h,r,3,i)[-1]
            diff = np.abs(r1-r2)
            if np.any(diff==0):
                diff = h*ep0
            rho = 30*h*ep0/diff
            rhomax = max(rho)
            if np.all(rho>1):
                t += 2*h
                track  = np.append(track,[r1],axis=0)
                p = np.power(min(rho),0.25)
                if p>2:
                    p = 2
                h = h*p
                if 2*h>T-t:
                    h = 0.5*(T-t)
                r = r1
                count = 0
            else:
                p = np.power(min(rho),0.25)
                h = h*p
                count += 1
    return track


def orbitalgr(N,pers):        #plots the GR effected orbits
    period = 15.559
    T = period*pers
    h = T/N
    e = 0.880
    a = 7
    rs = 2.0/63200
    r = np.zeros(4)
    r[0] = -a*(1+e)
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
    ax1[0].set_title("t = "+str(T))
    ax1[0].set_xlabel('step')
    ax1[0].set_ylabel('E')
    ax1[1].set_xlabel('x (AU)')
    ax1[1].set_ylabel('y (AU)')
    return np.abs(rad1[-1]-rad2[-1])
