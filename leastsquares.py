from __future__ import division
import numpy as np
from pylab import scatter,imshow,show,plot,xlabel,ylabel,xlim,ylim,suptitle

def leastsquares(x,y,tol):
  N = len(x)
  i = 0
  while i < N:
    if np.all(y[i:N] <= tol):
      b = y[0:i+1]
      i = N+1
    i+=1
  if i == N:
    b = y
  y = b
  x = x[:len(y)]
  Ex = np.sum(x)/N                                                       
  Ey = np.sum(y)/N
  Exx = np.sum(x*x)/N
  Exy = np.sum(x*y)/N
  m = (Exy-Ex*Ey)/(Exx-Ex*Ex)                                             
  c = (Exx*Ey-Ex*Exy)/(Exx-Ex*Ex)
  return m,c

