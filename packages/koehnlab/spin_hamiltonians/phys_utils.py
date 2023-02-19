import numpy as np
from .phys_const import kBcm

""" some general physics routines """


def getBoltzmannFactors(Elevels, Temp):
    """ compute the Boltzmann factors for all levels on Elevels (cm-1) for Temperature Temp (K) """
    beta = 1.0 / (kBcm * Temp)
    Factors = np.exp(-Elevels * beta)

    return Factors


def CLorentz(x, dlt):
    """ correlation function for Lorentz line shape """
    cl = 4.0 * np.exp(-dlt * x)
    return cl


def CGauss(x, sig):
    """ correlation function for Gauss line shape """
    cg = 4.0 * np.exp(-0.5 * x * x * sig * sig)
    return cg
