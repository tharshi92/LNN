# Coded by Tharshi Srikannathasan March 2016
# This is a 4th order Runge-Kutta Scheme to Solve the Lorentz Model

# import all needed libraries 
import numpy as np;
import pylab as pl;


# define simulation parameters
T = 8.0;    # simultion length
h = 1e-2;    # timestep
N = T/h;   # number of steps

params = np.array([10.0, 8.0/3.0, 28.0]);
x0 = np.array([1.508870, -1.531271, 25.46091]);

x = np.zeros(N);
y = np.zeros(N);
z = np.zeros(N);
dX = np.zeros(N);
dY = np.zeros(N);
dZ = np.zeros(N);

# parameters for the lorentz model
sigma = params[0];
beta = params[1];
rho = params[2];

# initial conditions
x[0] = x0[0];
y[0] = x0[1];
z[0] = x0[2];

# define dynamical equation

def dynamics(x, y, z, sigma, beta, rho):
    
    dxdt = -sigma*(x - y);
    
    dydt = x*(rho - z) - y;
    
    dzdt = x*y - beta*z;
    
    return dxdt, dydt, dzdt;

# integrate using Runge Kutta Method

for i in xrange(0, int(N - 1)):

    p1, q1, r1 = dynamics(x[i], y[i], z[i], sigma, beta, rho);
    dX[i] = p1;
    dY[i] = q1;
    dZ[i] = r1;

    p2, q2, r2 = dynamics(x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0,\
                          sigma, beta, rho);
    p3, q3, r3 = dynamics(x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0,\
                          sigma, beta, rho);
    p4, q4, r4 = dynamics(x[i] + h*p3, y[i] + h*q3, z[i] + h*r3,\
                          sigma, beta, rho);

    x[i+1] = x[i] + h*(p1 + 2.0*p2 + 2.0*p3 + p4)/6.0;
    y[i+1] = y[i] + h*(q1 + 2.0*q2 + 2.0*q3 + q4)/6.0;
    z[i+1] = z[i] + h*(r1 + 2.0*r2 + 2.0*r3 + r4)/6.0;

# write data to file

outFile = open("data.txt", "w");
outFile.write('x\ty\tz\txdot\tydot\tzdot\n');

for i in xrange(0, int(N - 1)):
    outFile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(x[i], y[i], z[i], dX[i], dY[i], dZ[i]));

print 'data exported to "data.txt"'



##fig1 = pl.figure();
##pl.plot(x, label='x');
##pl.plot(y, label='y');
##pl.plot(z, label='z');
##pl.plot(truth, label='truth'); 
##pl.legend();
##pl.xlabel("timestep");

##fig2 = pl.figure();
##ax = fig2.add_subplot(111, projection='3d');
##ax.scatter(x, y, z);
##
##ax.set_xlabel('X');
##ax.set_ylabel('Y');
##ax.set_zlabel('Z');

pl.show()


