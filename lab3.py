import numpy as np
from scipy.sparse import diags


def generate_safe_system(n):
    """
    Generate a linear system A x = b where A is strictly diagonally dominant,
    ensuring LU factorization without pivoting will work.

    Parameters:
        n (int): Size of the system (n x n)

    Returns:
        A (ndarray): n x n strictly diagonally dominant matrix
        b (ndarray): RHS vector
        x_true (ndarray): The true solution vector
    """

    k = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()

    # Solution is always all ones
    x_true = np.ones((n, 1))

    # Compute b = A @ x_true
    b = A @ x_true

    return A, b, x_true


def lu_factorisation(A):
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")
    
    L, U = np.zeros_like(A), np.zeros_like(A)

    # 1s in diagonal
    for i in range(n):
        L[i,i] = 1

    for j in range(n):
        # j th row in U from 0 to j
        for i in range(j + 1):
            sum = 0.0
            for k in range(i):
                sum += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum
        
        # j th row in L from j+1 to last row
        for i in range(j + 1, n):
            sum = 0.0
            for k in range(j):
                sum += L[i, k] * U[k, j]
            L[i, j] = (A[i, j] - sum) / U[j, j]
    
    return L, U


def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U


def forward_substitution(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        temp = b[i]
        for j in range(i-1):
            temp -= L[i,j] * x[j]
        x[i] = temp / L[i,i]
    return x


def backward_substitution(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        temp = b[i]
        for j in range(i+1, n):
            temp -= U[i,j] * x[j]
        x[i] = temp / U[i,i]
    return x


sizes = [2**j for j in range(1, 6)]

for n in sizes:
    # generate a random system of linear equations of size n
    A, b, x = generate_safe_system(n)

    # do the solve
    L, U = lu_factorisation(A)
    y = forward_substitution(L, b)
    solve = backward_substitution(U, y)

    print(f"Solve {n}: ", solve)

