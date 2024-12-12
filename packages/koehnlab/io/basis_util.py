import math

import numpy as np

from koehnlab.spin_hamiltonians import spin_utils

from .matrix_type import MatrixType


def get_propmat_prod(prop, row_mult: int, col_mult: int):
    """
    Returns the given (spatial) property matrix in product basis (spin-spatial basis).
    Args:
    ---------------------
    prop -- Property matrix in the basis of 0-th order wavefunctions
    row_mult -- Multiplicity of the states in the row of property matrix
    col_mult -- Multiplicity of the states in the cols of the property matrix
    Returns:
    ---------------------
    The property matrix transformed into the corresponding product basis
    """

    dist_mat = np.zeros(shape=(row_mult, col_mult))

    min_mult = min(row_mult, col_mult)
    row_offset = row_mult - min_mult
    col_offset = col_mult - min_mult

    # We assume that the associated property is independent of spin, meaning that states with different
    # M_S are orthogonal to each other. Therefore, we have to duplicate the prop matrix into the blocks
    # of the results which correspond to sub-blocks with identical M_S values.
    dist_mat[row_offset : row_offset + min_mult, col_offset : col_offset + min_mult] = (
        np.identity(min_mult)
    )

    return np.kron(dist_mat, prop)


def get_spinmat_prod(spin_mat, row_states: int, col_states: int):
    """
    Computes the spin matrix elements of the given spin matrix in the productbasis (spin-spatial basis).
    Blocked over ms number.
    Args:
    ------------------------
    sMat -- spin matrix in spin basis
    multiplicity -- Multiplicity of the system
    spat_num -- number of spatial states

    Returns:
    ------------------------
    SmatX,SmatY,SmatZ -- spin matrix in the productbasis
    """

    dist_mat = np.eye(row_states, col_states)

    kron_result = np.kron(spin_mat, dist_mat)
    return kron_result

    multiplicity = len(spin_mat) 

    dim = int(multiplicity * spat_num)
    spin = 0.5 * (multiplicity - 1)
    Smat = np.zeros((dim, dim), dtype=complex)
    if spin == 0:
        return Smat
    else:
        for x in range(dim):
            for y in range(dim):
                ms = math.floor(x / spat_num)
                ms_strich = math.floor(y / spat_num)
                i = x % spat_num
                j = y % spat_num
                if i == j:
                    Smat[x, y] = spin_mat[ms, ms_strich]
    return Smat


def transform_to_product_basis(matrix, spin_qns, state_nums, matrix_type: MatrixType):
    """
    Transforms the given matrix into the product basis

    Args:
    --------------------------
    matrix -- The matrix to transform
    spin_qns -- Array of spin quantum numbers (one entry per state group)
    state_nums -- Array with the amount of spatial states per state group (one entry per state group)
    matrix_type -- The type of the input matrix (i.e. whether it is a spin-like matrix or a spatial property matrix)

    Returns:
    --------------------------
    Matrix in the product basis
    """
    ngroups = len(spin_qns)
    assert ngroups == len(state_nums)

    multiplicities = [int(2 * S) + 1 for S in spin_qns]

    assert len(matrix) == sum(multiplicities) or len(matrix) == sum(state_nums)

    out_group_dims = multiplicities * state_nums
    dim = int(np.sum(out_group_dims))

    transformed = np.zeros((dim, dim), dtype=matrix.dtype)

    if matrix_type == MatrixType.Spatial:
        data_group_dims = state_nums
    else:
        assert matrix_type == MatrixType.Spin
        data_group_dims = multiplicities

    for i in range(ngroups):
        out_skip_rows = int(np.sum(out_group_dims[:i]))
        out_row_slice = (out_skip_rows, out_skip_rows + out_group_dims[i])

        data_skip_rows = int(np.sum(data_group_dims[:i]))
        data_row_slice = (data_skip_rows, data_skip_rows + data_group_dims[i])
        for j in range(ngroups):
            out_skip_cols = int(np.sum(out_group_dims[:j]))
            out_col_slice = (out_skip_cols, out_skip_cols + out_group_dims[j])

            data_skip_cols = int(np.sum(data_group_dims[:j]))
            data_col_slice = (data_skip_cols, data_skip_cols + data_group_dims[j])

            data_slab = matrix[
                data_row_slice[0] : data_row_slice[1],
                data_col_slice[0] : data_col_slice[1],
            ]

            if matrix_type == MatrixType.Spatial:
                prod_data = get_propmat_prod(
                    data_slab, row_mult=multiplicities[i], col_mult=multiplicities[j]
                )
            else:
                assert matrix_type == MatrixType.Spin
                prod_data = get_spinmat_prod(
                    data_slab, row_states=state_nums[i], col_states=state_nums[j]
                )

            transformed[
                out_row_slice[0] : out_row_slice[1], out_col_slice[0] : out_col_slice[1]
            ] = prod_data

    return transformed


def similarity_transform(matrix, U, unitary: bool = True):
    """
    Performs a similarity transformation of the given matrix

    Args:
    ---------------------
    matrix -- The matrix to transform
    U -- The matrix to use for the similarity transformation
    unitary -- Whether U is a unitary matrix (this is assumed by default)

    Returns:
    --------------------
    The transformed matrix
    """
    if unitary:
        return np.conj(U.T) @ matrix @ U

    return np.linalg.inv(U) @ matrix @ U
