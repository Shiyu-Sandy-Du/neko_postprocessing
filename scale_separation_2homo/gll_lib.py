#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:00:44 2023

@author: shiyud
copy from
https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=4SJ15xMoBUuG
and from /home/mech/johan/matlab/nekpost
"""


import numpy as np


def lgP (n, xi):
  """
  Evaluates P_{n}(xi) using an iterative algorithm
  """
  if n == 0:
    
    return np.ones (xi.size)
  
  elif n == 1:
    
    return xi

  else:

    fP = np.ones (xi.size); sP = xi.copy (); nP = np.empty (xi.size)

    for i in range (2, n + 1):

      nP = ((2.0 * i - 1.0) * xi * sP - (i - 1.0) * fP) / i

      fP = sP; sP = nP

    return nP

# def dLgP (n, xi):
#   """
#   Evaluates the first derivative of P_{n}(xi)
#   """
#   return n * (lgP (n - 1, xi) - xi * lgP (n, xi))\
#            / (1 - xi ** 2)

def dLgP (n, xi):
    dL = np.zeros(n+1)
    nc = 0
    j = 3
    dL[0] = 0.0
    dL[1] = 1.0

    for j in range(2,n+1):
        nc = j-1
        tmp = lgP(nc,xi)
        dL[j] = (2.0*nc+1.0)*tmp + dL[j-2]
    return dL[n]

def d2LgP (n, xi):
  """
  Evaluates the second derivative of P_{n}(xi)
  """
  return (2 * xi * dLgP (n, xi) - n * (n + 1)\
                                    * lgP (n, xi)) / (1 - xi ** 2)

def d3LgP (n, xi):
  """
  Evaluates the third derivative of P_{n}(xi)
  """
  return (4 * xi * d2LgP (n, xi)\
                 - (n * (n + 1) - 2) * dLgP (n, xi)) / (1 - xi ** 2)
      
def gLLNodesAndWeights (n, epsilon = 1e-15):
  """
  Computes the GLL nodes and weights
  """
  if n < 2:
    
    print ('Error: n must be larger than 1')
  
  else:
    
    x = np.empty (n)
    w = np.empty (n)
    
    x[0] = -1.0; x[n - 1] = 1.0
    w[0] = w[0] = 2.0 / ((n * (n - 1))); w[n - 1] = w[0];
    
    n_2 = n // 2
    
    for i in range (1, n_2):
      
      xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) *\
           np.cos ((4 * i + 1) * np.pi / (4 * (n - 1) + 1))
      
      error = 1.0
      
      while error > epsilon:
        "Evaluating GLL points by Halley's method"
        y  =  dLgP (n - 1, xi)
        y1 = d2LgP (n - 1, xi)
        y2 = d3LgP (n - 1, xi)
        
        dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)
        
        xi -= dx
        error = abs (dx)
      
      x[i] = -xi
      x[n - i - 1] =  xi
      
      w[i] = 2 / (n * (n - 1) * lgP (n - 1, x[i]) ** 2)
      w[n - i - 1] = w[i]

    if n % 2 != 0:

      x[n_2] = 0
      w[n_2] = 2.0 / ((n * (n - 1)) * lgP (n - 1, np.array (x[n_2])) ** 2)
      
  return x, w

def integrate_1d_using_gll_n(f, n_x):
    weights_x = gLLNodesAndWeights(n_x)[1]
    # integral = 0.0
    # for i in range(n_x):
    #     integral += f[i] * weights_x[i]
    integral = np.dot(f,weights_x)

    return integral

def integrate_1d_using_gll_w(f, weights_x):
    # integral = 0.0
    # for i in range(len(weights_x)):
    #     integral += f[i] * weights_x[i]
    integral = np.dot(f,weights_x)

    return integral

def integrate_2d_using_gll_n(f, n_x, n_y):
    weights_x = gLLNodesAndWeights(n_x)[1]
    weights_y = gLLNodesAndWeights(n_y)[1]
    integral = 0.0
    for i in range(n_x):
        for j in range(n_y):
            integral += f[i, j] * weights_x[i] * weights_y[j]

    return integral

def integrate_2d_using_gll_w(f, weights_x,  weights_y):
    integral = 0.0
    for i in range(len(weights_x)):
        for j in range(len(weights_y)):
            integral += f[i, j] * weights_x[i] * weights_y[j]

    return integral

def integrate_3d_using_gll_n(f, n_x, n_y, n_z):
    weights_x = gLLNodesAndWeights(n_x)[1]
    weights_y = gLLNodesAndWeights(n_y)[1]
    weights_z = gLLNodesAndWeights(n_z)[1]
    integral = 0.0
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                integral += f[i, j, k] * weights_x[i] * weights_y[j] *  weights_z[k]

    return integral

def integrate_3d_using_gll_w(f, weights_x, weights_y, weights_z):
    integral = 0.0
    for i in range(len(weights_x)):
        for j in range(len(weights_y)):
            for k in range(len(weights_z)):
                integral += f[i, j, k] * weights_x[i] * weights_y[j] * weights_z[k]

    return integral

def interp_1D_weights(ksi_N, N, x_hat):
    # suppose the function is defined on [-1,1]
    # f_gll -- field upon gll points
    # N -- order of polynomial
    # ksi_N -- gll defined on [-1,1]
    # x_hat -- target coordinates defined on [-1,1]

    dL_xhat = dLgP(N, x_hat)
    phi_N = np.zeros(N+1)
    for i_node in range(N+1):
        L_ksiN = lgP(N, ksi_N[i_node])
        if (N*(N+1)*(x_hat-ksi_N[i_node])*L_ksiN) == 0:
            phi_N[i_node] = 1.0
        else: 
            phi_N[i_node] = -(1-x_hat**2)*dL_xhat/\
                            (N*(N+1)*(x_hat-ksi_N[i_node])*L_ksiN)
      
    return phi_N

def lgPx_mat(N, x):
    # form a matrix PHI_N(x) with
    # PHI_N(i,j) = lgP_i(x_j)
    PHI_N = np.zeros([len(x),N+1])
    for i_x in range(len(x)):
        for i_order in range(N+1):
            PHI_N[i_x,i_order] = lgP(i_order,x[i_x]) 

    return PHI_N
    