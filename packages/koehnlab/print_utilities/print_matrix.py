# a few convenient output routines


def printMatC(mat, formatstr="7.2f", maxcol=10, put_labels=True, triangle=False):
    """ Print a matrix of complex values
       mat        --  a matrix (as exception also a vector is accepted)
       formatstr  -- format the real and imaginary part using this format (valid python format required)
       maxcol     -- max. number of columns in each output batch
       put_labels -- output the labels for rows and columns 
       triangle   -- only output the (lower) triangle
    """

    d1 = mat.shape[0]
    try:
        d2 = mat.shape[1]
    except:
        d2 = 1
    fmt = " {:" + formatstr + "}{:" + formatstr + "}  "
    spc = formatstr.split(".")[0]
    fmtspc = " {:" + spc + "}{:" + spc + "}  "
    bst = 0
    bnd = min(maxcol, d2)
    while bst < d2:
        if put_labels:
            print("{:5}".format(""), end="")
            for jj in range(bst, bnd):
                print(fmtspc.format(jj, " "), end="")
            print("")
        for ii in range(d1):
            if not triangle or ii >= bst:
                if put_labels:
                    print("{:5}".format(ii), end="")
                bnd2 = bnd
                if triangle:
                    bnd2 = min(bnd, ii + 1)
                for jj in range(bst, bnd2):
                    try:
                        print(fmt.format(mat[ii, jj].real, mat[ii, jj].imag), end="")
                    except:
                        print(fmt.format(mat[ii].real, mat[ii].imag), end="")
                print("")
        bst = bnd
        bnd = min(bst + maxcol, d2)
        print("")


def printMatF(mat, formatstr="12.6f", maxcol=10, put_labels=True, triangle=False):
    """ Print a matrix of float values
       mat        --  a matrix (as exception also a vector is accepted)
       formatstr  -- format the real and imaginary part using this format (valid python format required)
       maxcol     -- max. number of columns in each output batch
       put_labels -- output the labels for rows and columns 
       triangle   -- only output the (lower) triangle
    """

    d1 = mat.shape[0]
    try:
        d2 = mat.shape[1]
    except:
        d2 = 1
    fmt = " {:" + formatstr + "}  "
    spc = formatstr.split(".")[0]
    fmtspc = " {:" + spc + "}  "
    bst = 0
    bnd = min(maxcol, d2)
    while bst < d2:
        if put_labels:
            print("{:5}".format(""), end="")
            for jj in range(bst, bnd):
                print(fmtspc.format(jj), end="")
            print("")
        for ii in range(d1):
            if not triangle or ii >= bst:
                if put_labels:
                    print("{:5}".format(ii), end="")
                bnd2 = bnd
                if triangle:
                    bnd2 = min(bnd, ii + 1)
                for jj in range(bst, bnd2):
                    try:
                        print(fmt.format(mat[ii, jj]), end="")
                    except:
                        print(fmt.format(mat[ii]), end="")
                print("")
        bst = bnd
        bnd = min(bst + maxcol, d2)
        print("")
