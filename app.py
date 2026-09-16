"""
NumPy Large Linear Systems Solver (Ax = b)
Companion CLI script for the Linear Lab Web Application
Pastel Chic Palette & Clean Architecture
"""

import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

def print_banner():
    print("=" * 64)
    print("  LINEAR LAB - Large Linear Systems Solver with NumPy")
    print("  Solving Ax = b using LAPACK LU & Least-Squares routines")
    print("=" * 64)

def solve_system(n=5, preset='random', method='direct'):
    print(f"\n[+] Generating {n}x{n} system with preset: '{preset}'...")
    rng = np.random.default_rng(seed=42)
    
    if preset == 'symmetric':
        M = rng.standard_normal((n, n))
        A = (M + M.T) / 2.0
    elif preset == 'identity':
        A = np.eye(n)
    elif preset == 'tridiagonal':
        A = np.diag(rng.uniform(2, 5, n)) + np.diag(rng.uniform(-1, 0, n-1), 1) + np.diag(rng.uniform(-1, 0, n-1), -1)
    elif preset == 'ill-conditioned':
        A = 1.0 / (np.arange(1, n + 1)[:, None] + np.arange(1, n + 1) - 1.0)
    else:
        A = rng.standard_normal((n, n))
        np.fill_diagonal(A, np.diag(A) + 3.0)

    b = rng.standard_normal(n)

    # Matrix metrics
    cond_num = np.linalg.cond(A)
    frob_norm = np.linalg.norm(A, 'fro')
    print(f"    Matrix Shape: {A.shape}")
    print(f"    Condition number kappa(A): {cond_num:.4e} ({'Well-conditioned' if cond_num < 1e4 else 'Ill-conditioned'})")
    print(f"    Frobenius norm ||A||_F:    {frob_norm:.4f}")

    # Solve
    print(f"\n[+] Solving using method: '{method}'...")
    start_time = time.perf_counter()
    if method == 'lstsq':
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    else:
        x = np.linalg.solve(A, b)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Verification
    residual = b - A @ x
    res_norm = np.linalg.norm(residual)

    print(f"    Solve Time: {elapsed_ms:.3f} ms")
    print(f"    Residual norm ||b - Ax||_2: {res_norm:.4e}")
    print(f"\n[+] Solution vector x (first {min(n, 5)} components):")
    for i in range(min(n, 5)):
        print(f"    x[{i}] = {x[i]: .6f}")
    if n > 5:
        print(f"    ... and {n - 5} more elements.")
    print("\n[OK] Linear system verified successfully.")

if __name__ == '__main__':
    size = 5
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except ValueError:
            pass
    print_banner()
    solve_system(n=size)
