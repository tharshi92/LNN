# Coded by Tharshi Srikannathasan Feb 2016
# This is a 4th order Runge-Kutta Scheme to Solve the Lorentz Model

# import all needed libraries 
import numpy as np;
import pylab as pl;

# define simulation parameters
T = 4.0;    # simultion length
h = 0.001;    # number of steps
N = T/h;    # timestep

x = np.zeros(N);
y = np.zeros(N);
z = np.zeros(N);

# parameters for the lorentz model
sigma = 10.0;
beta = 8.0/3.0;
rho = 28;

# initial conditions
x[0] = -4.5;
y[0] = 1.7;
z[0] = 0.1;

# define derivative functions

def x_prime(sigma, x, y, z):
    
    return -sigma*(x - y);
    
def y_prime(rho, x, y, z):
    
    return x*(rho - z) - y;
    
def z_prime(beta, x, y, z):
    
    return x*y - beta*z;
    
# integrate using Runge Kutta Method

for i in xrange(0, int(N - 1)):

    p1 = x_prime(sigma, x[i], y[i], z[i]);
    q1 = y_prime(rho, x[i], y[i], z[i]); 
    r1 = z_prime(beta, x[i], y[i], z[i]);

    p2 = x_prime(sigma, x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0);
    q2 = y_prime(rho, x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0);
    r2 = z_prime(beta, x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0); 

    p3 = h*x_prime(sigma, x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0); 
    q3 = h*y_prime(rho, x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0); 
    r3 = h*z_prime(beta, x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0); 

    p4 = x_prime(sigma, x[i] + h*p3, y[i] + h*q3, z[i] + h*r3);
    q4 = y_prime(rho, x[i] + h*p3, y[i] + h*q3, z[i] + h*r3); 
    r4 = z_prime(beta, x[i] + h*p3, y[i] + h*q3, z[i] + h*r3);

    x[i+1] = x[i] + h*(p1 + 2.0*p2 + 2.0*p3 + p4)/6.0;
    y[i+1] = y[i] + h*(q1 + 2.0*q2 + 2.0*q3 + q4)/6.0;
    z[i+1] = z[i] + h*(r1 + 2.0*r2 + 2.0*r3 + r4)/6.0;
    
# plot results

pl.plot(x)
pl.plot(y)
pl.plot(z)
pl.xlabel("timestep")
pl.show()
    
print("Everything is Okay.")
