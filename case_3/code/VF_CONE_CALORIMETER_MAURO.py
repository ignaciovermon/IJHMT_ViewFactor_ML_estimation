#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:56:12 2025

@author: pablopintomoncada
"""

import numpy as np
from scipy.integrate import dblquad

# View factor from a differential element, dA, to the cone heater (helical coil - HC)
# Parameters:
# h: x-coordinate location (in mm)
# k: y-coordinate location (in mm)
# d: vertical distance (d1) from the lower of the cone heater base plate to the target (in mm)
# p: differential element size (diameter, in mm)

def VF_dA_to_HC(h, k, d, p):
    # Constants (Refer to ISO 5660-1 or ASTM E1354)
    R1 = 40
    R2 = 40
    H  = 65

    # Define the integral functions
    def f1(u, v):
        return (1/(np.pi**2*p**2)) * np.log(
            (0.5*p*np.cos(u)+h-(-R1-R2+R2*v/(19*np.pi))*np.cos(v))**2 +
            (0.5*p*np.sin(u)+k-(-R1-R2+R2*v/(19*np.pi))*np.sin(v))**2 +
            (H*v/(19*np.pi)+d)**2  ) * (p*R2/(38*np.pi)*np.sin(v-u)+0.5*p*(-R1-R2+(R2*v)/(19*np.pi))*np.cos(v-u)) 
    
    def f2(u, v):
        return (1/(np.pi**2*p**2)) * np.log(
            (0.5*p*np.cos(u)+h-(-R1-R2+R2*v/(19*np.pi))*np.cos(v))**2 +
            (0.5*p*np.sin(u)+k-(-R1-R2+R2*v/(19*np.pi))*np.sin(v))**2 +
            (H*v/(19*np.pi)+d)**2  ) * (p*R2/(38*np.pi)*np.sin(v-u)+0.5*p*(-R1-R2+(R2*v)/(19*np.pi))*np.cos(v-u)) 
    
    def f3(u, v):
        return (1/(np.pi**2*p**2)) * np.log(
            (0.5*p*np.cos(u)+h-(np.sqrt(H**2 + R2**2)/19)*np.cos(v)+R1+18*R2/19)**2 + 
            (0.5*p*np.sin(u)+k)**2 +
            ((np.sqrt(H**2 + R2**2)/19)*np.sin(v)+H/19 + d)**2)*(p/38)*np.sqrt(H**2 + R2**2)*np.sin(u)*np.sin(v)
    
    def f4(u, v):
        return (1/(np.pi**2*p**2)) * np.log(
            (0.5*p*np.cos(u)+h-(np.sqrt(H**2 + R2**2)/19)*np.cos(v)-R1-R2/19)**2 + 
            (0.5*p*np.sin(u)+k)**2 +
            ((np.sqrt(H**2 + R2**2)/19)*np.sin(v)+18*H/19+d)**2)*(p/38)*np.sqrt(H**2 + R2**2)*np.sin(v)*np.sin(u)
    
    # Compute the view factors
    VF = (dblquad(f1, 17*np.pi, 0, 0, 2*np.pi)[0] +
          dblquad(f2, 2*np.pi, 19*np.pi, 0, 2*np.pi)[0] +
          dblquad(f3, np.arctan(H/R2) - np.pi, np.arctan(H/R2), 0, 2*np.pi)[0] +
          dblquad(f4, np.pi - np.arctan(H/R2), 2*np.pi - np.arctan(H/R2), 0, 2*np.pi)[0])
    
    return VF

# Example usage
h = 0
k = 0
d = 30
p = 1

VF = VF_dA_to_HC(h,k,d,p)
print(f'The View Factor is: {VF:.4f}')