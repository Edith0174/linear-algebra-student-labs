def lu_factorisation(A):
    n, m = A.shape
    if n != m:
        raise ValueError(f"A is not square")
    
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