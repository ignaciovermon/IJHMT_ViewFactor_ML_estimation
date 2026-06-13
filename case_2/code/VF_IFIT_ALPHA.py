#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 12:08:53 2025

@author: pablopintomoncada
"""

import numpy as np
from scipy.integrate import dblquad

# Parámetros
H = 46
d = 21.2
R = 5
p = 0.1
alpha = np.arctan(5 / np.sqrt(21.2**2 - 5**2))

def f(b):
    # Primer término
    def integrand1(u, v):
        return (np.log((0.5*p*np.cos(u) - R*np.cos(v))**2 +
                       (d - R*np.sin(v))**2 +
                       (0.5*p*np.sin(u) + b + 0.5*p)**2)
                * 0.5*p*R*np.sin(u)*np.sin(v))
    term1, _ = dblquad(integrand1, alpha, np.pi - alpha, lambda v: 0, lambda v: 2*np.pi)

    # Segundo término
    def integrand2(u, v):
        return (np.log((0.5*p*np.cos(u) + R*np.cos(alpha))**2 +
                       (d - R*np.sin(alpha))**2 +
                       (0.5*p*np.sin(u) + b + 0.5*p - H*v)**2)
                * 0.5*p*H*np.cos(u))
    term2, _ = dblquad(integrand2, 0, 1, lambda u: 0, lambda u: 2*np.pi)

    # Tercer término
    def integrand3(u, v):
        return (np.log((0.5*p*np.cos(u) - R*np.cos(v))**2 +
                       (d - R*np.sin(v))**2 +
                       (0.5*p*np.sin(u) + b + 0.5*p - H)**2)
                * 0.5*p*R*np.sin(u)*np.sin(v))
    term3, _ = dblquad(integrand3, np.pi - alpha, alpha, lambda v: 0, lambda v: 2*np.pi)

    # Cuarto término
    def integrand4(u, v):
        return (np.log((0.5*p*np.cos(u) - R*np.cos(alpha))**2 +
                       (d - R*np.sin(alpha))**2 +
                       (0.5*p*np.sin(u) + b + 0.5*p - H + H*v)**2)
                * 0.5*p*H*np.cos(u))
    term4, _ = dblquad(integrand4, 0, 1, lambda u: 0, lambda u: 2*np.pi)

    resultado = (1 / (2*np.pi * 2*np.pi * (p/2)**2)) * (term1 + term2 + term3 - term4)
    return resultado

# Ejemplo de uso
b_val = 6.1
resultado = f(b_val)
print("f(b=0) =", resultado)