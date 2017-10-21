from __future__ import division
import pylab as plt
import numpy as np
import math
import time
import leastsquares as ls
from methods import feul,rk2,rk4,verlet,df,tomin,secant,newraph,relax,bisect,adaprk4


def orbital(N,T):        #calculates orbit of mars and outputs graphs if need-be
    h = T/N
    a = 1.523679
    e = 0.0934
    r = np.zeros(4)
    r[0] = -a*(1+e)
    r[3] = 2*np.pi*np.sqrt((1-e)/(a*(1+e)))
    track1 = feul(h,r,N)   #does it for all 4 methods
    track2 = rk2(h,r,N)
    track3 = rk4(h,r,N)
    track4 = verlet(h,r,N)
    rad1 = np.sqrt(np.power(track1[:,0],2)+np.power(track1[:,1],2))
    E1 = .5*(np.power(track1[:,3],2)+np.power(track1[:,2],2))-4*np.power(np.pi,2)*np.power(rad1,-1)
    rad2 = np.sqrt(np.power(track2[:,0],2)+np.power(track2[:,1],2))
    E2 = .5*(np.power(track2[:,3],2)+np.power(track2[:,2],2))-4*np.power(np.pi,2)*np.power(rad2,-1)
    rad3 = np.sqrt(np.power(track3[:,0],2)+np.power(track3[:,1],2))
    E3 = .5*(np.power(track3[:,3],2)+np.power(track3[:,2],2))-4*np.power(np.pi,2)*np.power(rad3,-1)
    rad4 = np.sqrt(np.power(track4[:,0],2)+np.power(track4[:,1],2))
    E4 = .5*(np.power(track4[:,3],2)+np.power(track4[:,2],2))-4*np.power(np.pi,2)*np.power(rad4,-1)
#     fig1, ax1 = plt.subplots(2,1)
#     p1, = ax1[0].plot(E1,'r-',label = 'Forward Euler')
#     ax1[1].plot(track1[:,0],track1[:,1],'r-')
#     p2, = ax1[0].plot(E2,'g-',label = 'RK2' )
#     ax1[1].plot(track2[:,0],track2[:,1],'g-')
#     p3, = ax1[0].plot(E3,'b-',label = 'RK4')
#     ax1[1].plot(track3[:,0],track3[:,1],'b-')
#     p4, = ax1[0].plot(E4,'k-',label = 'Verlet')
#     ax1[1].plot(track4[:,0],track4[:,1],'k-')
#     l0 = ax1[0].legend(handles=[p1,p2,p3,p4],loc=3)
#     l1 = ax1[1].legend(handles=[p1,p2,p3,p4],loc=1)
#     ax1[0].set_title("t = "+str(T))
#     ax1[0].set_xlabel('step')
#     ax1[0].set_ylabel('E')
#     ax1[1].set_xlabel('x (AU)')
#     ax1[1].set_ylabel('y (AU)')    
    return np.array([rad1[-1],rad2[-1],rad3[-1],rad4[-1]])

def halley(N,pers):  #calculates orbit of Halley's comit for a certain number of periods
    period = 75
    T = pers*period
    h = T/N
    a = 17.834144
    e = .96714
    r = np.zeros(4)
    r[0] = -a*(1+e)
    r[3] = 2*np.pi*np.sqrt((1-e)/(a*(1+e)))
    t1 = time.time()
    track1 = rk4(h,r,N)          #does this for RK4, verlet and our adaptive program
    t2 = time.time()
    track2 = verlet(h,r,N)
    t3 = time.time()
    track3 = adaprk4(h,r,T)
    t4 = time.time()
    rad1 = np.sqrt(np.power(track1[:,0],2)+np.power(track1[:,1],2))
    rad2 = np.sqrt(np.power(track2[:,0],2)+np.power(track2[:,1],2))
    rad3 = np.sqrt(np.power(track3[:,0],2)+np.power(track3[:,1],2))
    E1 = .5*(np.power(track1[:,3],2)+np.power(track1[:,2],2))-np.power(rad1,-1)
    E2 = .5*(np.power(track2[:,3],2)+np.power(track2[:,2],2))-np.power(rad2,-1)
    E3 = .5*(np.power(track3[:,3],2)+np.power(track3[:,2],2))-np.power(rad3,-1)
    # fig1, ax1 = plt.subplots(2,1)
    # p1, = ax1[0].plot(E1,'r-',label = 'RK4')
    # ax1[1].plot(track1[:,0],track1[:,1],'r-')
    # p2, = ax1[0].plot(E2,'g-',label = 'Verlet' )
    # ax1[1].plot(track2[:,0],track2[:,1],'g-')
    # p3, = ax1[0].plot(E3,'b-',label = 'Adaptive RK4')
    # ax1[1].plot(track3[:,0],track3[:,1],'b-')
    # l0 = ax1[0].legend(handles=[p1,p2,p3],loc=1)
    # ax1[0].set_title("t = "+str(T))
    # ax1[0].set_xlabel('step')
    # ax1[0].set_ylabel('E')
    # ax1[1].set_xlabel('x (AU)')
    # ax1[1].set_ylabel('y (AU)')    
    # ax1[1].set_xlim([-40,5])
    # ax1[1].set_ylim([-7,7])
    print  (t2-t1,t3-t2,t4-t3)                       #outputs the time taken to evaluate
    return np.array([rad1[-1],rad2[-1],rad3[-1]])


def halleyerror(N,pers):
    period = 75            #gives convergence plots for halley's comet
    T = pers*period
    ap = 17.834144
    e = .96714
    terms = len(N)
    tol = 1e-12
    a = np.zeros([terms,3])
    exact = newraph(1000,T,ap,e)        #newton-raphson gives the exact answer
    for i in range(terms):
        h =  np.abs(halley(N[i],pers)-exact)
        a[i,:] = h  
    lN = np.log10(N)
    la = np.log10(a)
    least  = np.zeros([3,2])
    for j in range(3):
        least[j,:] = ls.leastsquares(lN,la[:,j],np.log10(.1*tol))
    
    fig1, ax1 = plt.subplots(1,1)    #plots everything
    ax1.plot(lN, la[:,0],'ko')
    p0, = ax1.plot(lN,least[0,0]*lN+least[0,1],'k-',label = '%.4f RK4' % least[0,0])
    ax1.set_title("t = "+str(T))
  
    ax1.plot(lN, la[:,1],'ro')
    p1, = ax1.plot(lN,least[1,0]*lN+least[1,1],'r-',label = '%.4f Verlet' % least[1,0])

    ax1.plot(lN, la[:,2],'bo')
    p2, = ax1.plot(lN,least[2,0]*lN+least[2,1],'b-',label = '%.4f Adaptive RK4' % least[2,0])
    ax1.set_xlabel("Log N")
    ax1.set_ylabel("Root Log Error")
    ax1.set_ylim([np.log10(0.1*tol),1])
    
    l0 = ax1.legend(handles=[p0,p1,p2],loc=1)
    return ax1

    


def errorplot(N,T,filename=None):
    terms = len(N)                   #convergenceplots for mars orbit
    a = np.zeros([terms,4])
    exact = newraph(1000,5,1.523679,0.0934)    #again NR gives exact answer
    for i in range(terms):
        for j in range(4):
            a[i,j] = np.abs(orbital(N[i],T)[j]-exact)
    lN = np.log10(N)
    la = np.log10(a)
    least  = np.zeros([4,2])
    tol = 1e-10
    for j in range(4):
        least[j,:] = ls.leastsquares(lN,la[:,j],np.log10(.1*tol))
    
    fig1, ax1 = plt.subplots(1,1)     #pretty plots
    ax1.plot(lN, la[:,0],'ko')
    p0, = ax1.plot(lN,least[0,0]*lN+least[0,1],'k-',label = '%.4f Forward Euler' % least[0,0])
    ax1.set_title("t = "+str(T))
  
    ax1.plot(lN, la[:,1],'ro')
    p1, = ax1.plot(lN,least[1,0]*lN+least[1,1],'r-',label = '%.4f RK2' % least[1,0])

    ax1.plot(lN, la[:,2],'bo')
    p2, = ax1.plot(lN,least[2,0]*lN+least[2,1],'b-',label = '%.4f RK4' % least[2,0])

    ax1.plot(lN, la[:,3],'go')
    p3, = ax1.plot(lN,least[3,0]*lN+least[3,1],'g-',label = '%.4f Verlet' % least[3,0])

    l0 = ax1.legend(handles=[p0,p1,p2,p3],loc=1)  

    # ax1[0].set_ylabel(r'Euler')
    # ax1[1].set_ylabel(r'RK2')
    # ax1[2].set_ylabel(r'RK4')
    # ax1[3].set_ylabel(r'Verlet')
    # ax1[3].set_xlabel(r'Log $N$')


def rooterror(N,T,filename=None):           #plots the convergence of root-finding methods
    terms = len(N)
    a = np.zeros([terms,3])
    exact = newraph(1000,5,1.523679,0.0934)    #again, thanks NR!
    tol = 1e-14
    for i in range(terms):
        a[i,0] = np.abs(secant(N[i],T)-exact)
        a[i,1] = np.abs(relax(N[i],T)-exact)
        a[i,2] = np.abs(bisect(N[i],T)-exact)
        for j in range(3):
            if a[i,j]<tol:
                a[i,j] = tol*0.1
    lN = N
    la = np.log10(a)
    least  = np.zeros([3,2])
    for j in range(3):
            least[j,:] = ls.leastsquares(lN,la[:,j],np.log10(tol))
    
    fig1, ax1 = plt.subplots(1,1)

    ax1.plot(lN, la[:,0],'ko')
    p0, = ax1.plot(lN,least[0,0]*lN+least[0,1],'k-',label = '%.4f Secant' % least[0,0])
    ax1.set_title("t = "+str(T))
  
    ax1.plot(lN, la[:,1],'ro')
    p1, = ax1.plot(lN,least[1,0]*lN+least[1,1],'r-',label = '%.4f Relaxation' % least[1,0])

    ax1.plot(lN, la[:,2],'bo')
    p2, = ax1.plot(lN,least[2,0]*lN+least[2,1],'b-',label = '%.4f Bisection' % least[2,0])
    ax1.set_xlabel("N")
    ax1.set_ylabel("Root Log Error")
    ax1.set_ylim([np.log10(0.1*tol),1])
    
    l0 = ax1.legend(handles=[p0,p1,p2],loc=1)  
