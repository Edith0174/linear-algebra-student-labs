import numpy as np


print(f"{'ε':<10} {'error1':<15} {'error2':<15} {'error3':<15}")
print("-" * 55)


def gram_schmidt_qr(A):
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        # Start with the j-th column of A
        u = A[:, j].copy()

        # Orthogonalize against previous q vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # projection coefficient
            u -= R[i, j] * Q[:, i]  # subtract the projection

        # Normalize u to get q_j
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R

for i in range(6, 17):
    eps = 10**(-i)
    
    A = np.array([[1, 1 + eps],
                  [1 + eps, 1]])
    
    Q, R = gram_schmidt_qr(A)
    
    error1 = np.linalg.norm(A - Q @ R)
    error2 = np.linalg.norm(Q.T @ Q - np.eye(2))
    error3 = np.linalg.norm(R - np.triu(R))
    
    print(f"{eps:<10.0e} {error1:<15.6e} {error2:<15.6e} {error3:<15.6e}")
