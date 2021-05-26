# -*- coding: utf-8 -*-
"""
Created on Sun May 23 21:21:50 2021

@author: Erdenetuya, Xuantong Pan
"""

import numpy as np
import matplotlib.pyplot as plt


# Auswertung des Polynoms mit Koeffizienten b und Stuetzstellen xk, an den Stellen x
def polNewtonHorner(bk, xk, x):
    """
    

    Parameters
    ----------
    bk : Koeffizienten bezüglich der Newtonbasis
    xk : Stuetzstellen
    x  : ndarray, von xk[0] bis xk[1]

    Returns
    -------
    y : interpolationspolynom

    """    
    """ Polynomonauswertung y = b[0] + b[1]*(x - xk[0]) + ... + b[n]*(x - x[0])* ... *(x - x[n-1]),
    mit Hornerschema """
    y = np.ones(x.shape) * bk[-1]
    for j in range(len(bk)-2, -1, -1):
        y = bk[j] + (x - xk[j])*y   
    return y

def koeffNewtonBasis(xk, yk):
    """
    

    Parameters
    ----------
    xk : Stuetzstellen
    yk : Stützwerten

    Returns
    -------
    der Koeffizienten bezüglich der Newtonbasis 

    """
    """ Berechnung mit dem rekursiven Schema der dividierten Differenzen - iterativ implementiert """
    m = len(xk)
    F = np.zeros((m,m)) # in der linken unteren Dreiecksmatrix werden alle berechneten 
                        # dividierten Differenzen gespeichtert, 
                        # auf der Diagonalen liegen dann die gesuchen Koeffizienten b_i
   
    F[:,0] = yk
    for j in range(1, m):     # j-te dividierte Diffenzen
        for i in range(j, m):
            F[i, j] = (F[i,j-1] - F[i-1,j-1])/(xk[i] - xk[i-j])
            
    return np.diag(F)

def poly_interpol(x, xi, yi):
    """
    

    Parameters
    ----------
    x  : ndarray, von xi[0] bis xi[1]
    xi : Stuetzstellen
    yi : Stützwerten

    Returns
    -------
    y : interpolationspolynom

    """
    
    bk  = koeffNewtonBasis(xi, yi)
    y = polNewtonHorner(bk, xi, x)
    
    return y


def splinej(x, j, xi, yi, mi):
    '''Auswertung des kubischen Polynoms s_j'''
    '''
    Parameters
    ----------
    x  : ndarray, von xi[0] bis xi[1]
    j  : Stelle von xi
    xi : Stuetzstellen 
    yi : Stützwerten
    mi : Lösungen von LGS für die zweiten Ableitungen 𝑚𝑗=𝑠′′(𝑥𝑗) , 𝑗=1,2, m[0] = m[-1] = 0
    
    Return
    -------
    y: interpolationspolynom

    '''
    h = xi[j+1] - xi[j]
    s = ( mi[j]*(xi[j+1] - x)**3 + mi[j+1]*(x - xi[j])**3 )/6/h  \
    + ( yi[j]*(xi[j+1] - x)    + yi[j+1]*(x - xi[j])    )/h      \
    -   ( mi[j]*(xi[j+1] - x)    + mi[j+1]*(x - xi[j])    )*h/6.
    return s


def splineinterpol(x, xi, yi):
    '''
    Parameters
    ----------
    x  : ndarray, von xi[0] bis xi[1]
    xi : Stuetzstellen 
    yi : Stützwerten
    
    Return
    -------
    y: interpolationspolynom

    '''

    n = xi.shape[0]
    
    # LGS for m_j, j = 1,..., n-1, which m_0 = m_n = 0
    A = np.eye(n-2)*4
    A += np.eye(n-2, k=1) + np.eye(n-2, k=-1)
    '''A = 
       [4, 1, 0, ....]
       [1, 4, 1, 0...]
       ...............
       [0,..,1 , 4, 1]
       [0, 0, 0, 1, 4]
    '''
    b = yi.copy()

    for i in range(1, n-1):
        b[i] = (yi[i-1] - 2*yi[i] + yi[i+1])*6/(xi[i]-xi[i-1])**2
    
    mi = xi.copy()
    mi[0] = mi[n-1] = 0
    
    mi[1:-1] = np.linalg.solve(A, b[1:-1])
    
    y = np.zeros(x.shape)
    for j in range(n-1):
        ind = np.nonzero((xi[j] <= x) & (x <= xi[j+1]))
        
        y[ind] = splinej(x[ind], j, xi, yi, mi)
    return y


def test_splineinterpol():
    '''
    test-function for splineinterpol()
    '''
    xi = np.array([-1, 0, 1, 2], dtype=float)
    yi = np.array([10, -2, 9, -4], dtype=float)
    x_0 = np.linspace(-1, 0, 100)
    x_1 = np.linspace(0, 1, 100)
    x_2 = np.linspace(1, 2, 100)
    
    s_0 = splineinterpol(x_0, xi ,yi)
    s_1 = splineinterpol(x_1, xi ,yi)
    s_2 = splineinterpol(x_2, xi ,yi)

    plt.figure(figsize=(8,4))
    plt.plot(xi, yi, '*', markersize=10)
    plt.plot(x_0, s_0, x_1, s_1, x_2, s_2, linewidth = 2);
    plt.legend(['', r'$s_0(x)$', r'$s_1(x)$', r'$s_2(x)$']);
    plt.xlabel('x')
    plt.ylabel('y')
    
    
def test_poly_interpol():
    '''
    test-function for splineinterpol()
    '''
    xi = np.array([-1, 0, 1, 2], dtype=float)
    yi = np.array([10, -2, 9, -4], dtype=float)
    x_0 = np.linspace(-1, 0, 100)
    x_1 = np.linspace(0, 1, 100)
    x_2 = np.linspace(1, 2, 100)
    
    s_0 = poly_interpol(x_0, xi ,yi)
    s_1 = poly_interpol(x_1, xi ,yi)
    s_2 = poly_interpol(x_2, xi ,yi)

    plt.figure(figsize=(8,4))
    plt.plot(xi, yi, '*', markersize=10)
    plt.plot(x_0, s_0, x_1, s_1, x_2, s_2, linewidth = 2);
    plt.legend(['', r'$s_0(x)$', r'$s_1(x)$', r'$s_2(x)$']);
    plt.xlabel('x')
    plt.ylabel('y')
    
    
if __name__ == "__main__":
    test_splineinterpol()
    test_poly_interpol()
