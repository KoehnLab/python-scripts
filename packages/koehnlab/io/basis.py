from enum import Enum

class Basis(Enum):
    # Basis of 0-th order wavefunctions
    WF0 = 0
    # Basis of products of WF0 and all spin-degrees of freedom
    Product = 1
    # Eigenbasis of the spin-orbit coupling matrix
    SpinOrbit = 2
