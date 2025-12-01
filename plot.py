import numpy as np
from scipy.sparse import diags
import time
import matplotlib.pyplot as plt


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
    """LU factorization without pivoting"""
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")
    
    L, U = np.zeros_like(A, dtype=float), np.zeros_like(A, dtype=float)

    # 1s in diagonal of L
    for i in range(n):
        L[i, i] = 1

    for j in range(n):
        # j-th column of U (rows 0 to j)
        for i in range(j + 1):
            sum_val = 0.0
            for k in range(i):
                sum_val += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum_val
        
        # j-th column of L (rows j+1 to n-1)
        for i in range(j + 1, n):
            sum_val = 0.0
            for k in range(j):
                sum_val += L[i, k] * U[k, j]
            L[i, j] = (A[i, j] - sum_val) / U[j, j]
    
    return L, U


def forward_substitution(L, b):
    """Solve Lx = b for x (forward substitution)"""
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        temp = b[i]
        for j in range(i):  # Fixed: was range(i-1)
            temp -= L[i, j] * x[j]
        x[i] = temp / L[i, i]
    return x


def backward_substitution(U, b):
    """Solve Ux = b for x (backward substitution)"""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        temp = b[i]
        for j in range(i + 1, n):
            temp -= U[i, j] * x[j]
        x[i] = temp / U[i, i]
    return x


def solve_with_lu(A, b):
    """Solve Ax = b using LU factorization"""
    L, U = lu_factorisation(A)
    y = forward_substitution(L, b.flatten())
    x = backward_substitution(U, y)
    return x


def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with back substitution.
    Standard implementation from lecture notes.
    """
    n = A.shape[0]
    # Create augmented matrix [A|b]
    Ab = np.hstack([A.copy(), b.copy()])
    
    # Forward elimination
    for k in range(n - 1):
        for i in range(k + 1, n):
            if Ab[k, k] == 0:
                raise ValueError("Zero pivot encountered")
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x


def time_solver(solver_func, A, b, name, num_runs=5):
    """Time a solver function over multiple runs"""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        x = solver_func(A, b)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, x


# Main comparison
print("=" * 80)
print("LU FACTORIZATION vs GAUSSIAN ELIMINATION PERFORMANCE COMPARISON")
print("=" * 80)

sizes = [2**j for j in range(1, 11)]  # Extended range for better comparison
lu_times = []
gauss_times = []

print(f"\n{'Size':<8} {'LU Time (ms)':<15} {'Gauss Time (ms)':<15}")
print("-" * 40)

for n in sizes:
    # Generate system
    A, b, x_true = generate_safe_system(n)
    
    # Time LU factorization method
    lu_time, lu_std, x_lu = time_solver(solve_with_lu, A, b, "LU", num_runs=5)
    
    # Time Gaussian elimination
    gauss_time, gauss_std, x_gauss = time_solver(gaussian_elimination, A, b, "Gauss", num_runs=5)
    
    # Store times for plotting
    lu_times.append(lu_time * 1000)  # Convert to ms
    gauss_times.append(gauss_time * 1000)
    
    print(f"{n:<8} {lu_time*1000:<15.4f} {gauss_time*1000:<15.4f}")

# Detailed breakdown for a specific size
print("\n" + "=" * 80)
print("DETAILED TIMING BREAKDOWN (n=256)")
print("=" * 80)

n = 256
A, b, x_true = generate_safe_system(n)

# Time each component of LU method
start = time.perf_counter()
L, U = lu_factorisation(A)
lu_factor_time = time.perf_counter() - start

start = time.perf_counter()
y = forward_substitution(L, b.flatten())
forward_time = time.perf_counter() - start

start = time.perf_counter()
x = backward_substitution(U, y)
backward_time = time.perf_counter() - start

total_lu_time = lu_factor_time + forward_time + backward_time

print(f"\nLU Factorization Method:")
print(f"  1. LU Factorization:      {lu_factor_time*1000:.4f} ms ({lu_factor_time/total_lu_time*100:.1f}%)")
print(f"  2. Forward Substitution:  {forward_time*1000:.4f} ms ({forward_time/total_lu_time*100:.1f}%)")
print(f"  3. Backward Substitution: {backward_time*1000:.4f} ms ({backward_time/total_lu_time*100:.1f}%)")
print(f"  Total:                    {total_lu_time*1000:.4f} ms")

# Time Gaussian elimination
start = time.perf_counter()
x_gauss = gaussian_elimination(A, b)
gauss_total_time = time.perf_counter() - start

print(f"\nGaussian Elimination:")
print(f"  Total:                    {gauss_total_time*1000:.4f} ms")

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(sizes, lu_times, 'o-', label='LU Factorization', linewidth=2, markersize=8)
plt.plot(sizes, gauss_times, 's-', label='Gaussian Elimination', linewidth=2, markersize=8)
plt.xlabel('Matrix Size (n)', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.title('LU Factorization vs Gaussian Elimination', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('solver_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'solver_comparison.png'")
plt.show()

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""
For this comparison:
- Both methods have O(n³) complexity for the main operations
- LU factorization: ~(2/3)n³ operations for factorization + 2n² for solving
- Gaussian elimination: ~(1/2)n³ operations total

The advantage of LU factorization becomes apparent when:
1. Solving multiple systems with the same A but different b vectors
   (factorization is done once, then O(n²) solves for each b)
2. Computing determinants or matrix inverses
3. When the structure allows for better numerical stability analysis

For a single solve, both methods perform similarly with the same complexity class.
""")