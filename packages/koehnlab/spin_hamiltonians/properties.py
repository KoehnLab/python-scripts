import numpy as np

from .phys_const import kBcm, ChiVVCGS


def getChiVV(En, MmatT, fBoltz, T):
    """compute chi*T by the van Vleck equation"""
    """ the chi() subroutine of Chibutaru and Ungur (SINGLE_ANISO) is acknowledged
        for inspiration and debugging of prefactors """

    ndim = En.shape[0]
    if ndim != MmatT.shape[1]:
        print("ChiVV: Error - conflicting dimensions for En and MmatT")
        exit(200)
    if ndim != fBoltz.shape[0]:
        print("ChiVV: Error - conflicting dimensions for En and fBoltz")
        exit(201)

    # print('start new ChiT(VV) computation:')

    ChiT = np.zeros((3, 3))

    for ii in range(ndim):
        if fBoltz[ii] > 1e-12:
            # print('state ',ii,' has sufficient occupation:',fBoltz[ii])

            for jj in range(ndim):
                denom = En[ii] - En[jj]
                if np.abs(denom) < 0.001:  # treat this as first-order contribution
                    # print('state ',jj,' considered as first order')
                    fac = fBoltz[ii]
                else:  # and the rest as second-oder contributions
                    # print('state ',jj,' considered as second order')
                    fac = -2.0 * fBoltz[ii] * kBcm * T / denom

                # print(' ic     jc       Mi        Mj')
                for ic in range(3):
                    for jc in range(3):
                        # print(ic,jc,MmatT[ic,ii,jj],MmatT[jc,jj,ii])
                        ChiT[ic, jc] += (
                            fac * (MmatT[ic, ii, jj] * MmatT[jc, jj, ii]).real
                        )

    Q = np.sum(fBoltz)
    ChiT *= 1.0 / Q * ChiVVCGS

    return ChiT
