from __future__ import print_function
from numpy import *
from numpy.linalg import *

def mvnpdf(x, mu, sigma):
    if not sigma.shape:
        sigma = reshape(sigma, (1, 1));
    if det(sigma) == 0:
        return 0;
    k = x.shape[0];
    dev = x - mu;
    inv_sigma = inv(sigma);
    desc = dot(dev, dot(inv_sigma, dev));
    c0 = (2*pi)**(-0.5*k);
    c1 = det(sigma)**-0.5;
    return c0*c1*exp(-0.5*desc);
