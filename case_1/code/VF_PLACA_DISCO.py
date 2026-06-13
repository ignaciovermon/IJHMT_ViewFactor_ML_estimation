#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:30:17 2025

@author: pablopintomoncada

d es la distancia (cateto) entre el elemento diferencial y la placa
h,k es la posición del elemento diferencial
"""

import numpy as np
from scipy.integrate import dblquad

# Parámetros
L = 100.0/2 # Largo del rectangulo
p = 80.0/2 # Ancho del rectangulo
R = 1e-2/2 # Radio del disco

def f(d, h, k):
    """
    Calcula la misma función definida en Mathematica:
      f[d_,h_,k_] := (1/(2π·2·p·L)) * (–I1 + I2 + I3 – I4)
    donde cada Ij es una integral doble sobre u∈[0,2π], v∈[0,1].
    """
    # Integrandos
    def integrand1(u, v):
        return np.log(
            (R*np.cos(u) + h - L/2)**2
          + (R*np.sin(u) + k - p/2 + p*v)**2
          + (-d)**2
        ) * (p * R * np.cos(u))

    def integrand2(u, v):
        return np.log(
            (R*np.cos(u) + h - L/2 + L*v)**2
          + (R*np.sin(u) + k + p/2)**2
          + (-d)**2
        ) * (L * R * np.sin(u))

    def integrand3(u, v):
        return np.log(
            (R*np.cos(u) + h + L/2)**2
          + (R*np.sin(u) + k + p/2 - p*v)**2
          + (-d)**2
        ) * (p * R * np.cos(u))

    def integrand4(u, v):
        return np.log(
            (R*np.cos(u) + h + L/2 - L*v)**2
          + (R*np.sin(u) + k - p/2)**2
          + (-d)**2
        ) * (L * R * np.sin(u))

    # Cálculo de cada integral doble (retorna tuple (valor, error))
    I1, _ = dblquad(integrand1,
                    0.0, 1.0,                  # v de 0 a 1
                    lambda v: 0.0,             # u desde 0
                    lambda v: 2*np.pi)         # hasta 2π

    I2, _ = dblquad(integrand2,
                    0.0, 1.0,
                    lambda v: 0.0,
                    lambda v: 2*np.pi)

    I3, _ = dblquad(integrand3,
                    0.0, 1.0,
                    lambda v: 0.0,
                    lambda v: 2*np.pi)

    I4, _ = dblquad(integrand4,
                    0.0, 1.0,
                    lambda v: 0.0,
                    lambda v: 2*np.pi)

    # Combinación final
    return (-I1 + I2 + I3 - I4) / (2 * np.pi * 2 * p * L)

# Ejemplo de uso:
if __name__ == "__main__":
    d_val, h_val, k_val = 80.0/2, 40.0/2, 40.0/2
    resultado = f(d_val, h_val, k_val)
    print(f"f({d_val}, {h_val}, {k_val}) =", resultado)