# Coded by Tharshi Srikannathasan March 2016
# This is the Lorentz Model dynamical equation

# import all needed libraries 
import numpy as np;

# define dynamical equation

def Lorentz(x, y, z):

    # parameters for the lorentz model
    sigma = 10.0;
    beta = 8.0/3.0;
    rho = 28;
    
    dxdt = -sigma*(x - y);
    
    dydt = x*(rho - z) - y;
    
    dzdt = x*y - beta*z;
    
    return dxdt, dydt, dzdt;