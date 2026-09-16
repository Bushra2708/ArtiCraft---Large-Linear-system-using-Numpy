import type { Matrix, Vector, SolverResult, IterationStep } from './types';
import { cloneMatrix, createVector, matrixVectorMultiply, vectorSubtract, vectorNorm } from './matrix';

/**
 * Solves Ax = b using Gaussian Elimination with Partial Pivoting (equivalent to NumPy np.linalg.solve)
 */
export function solveDirectLU(A: Matrix, b: Vector): SolverResult {
  const startTime = performance.now();
  const n = A.length;
  const cols = A[0]?.length || 0;

  if (n !== cols) {
    // If not square, fallback to Least Squares
    return solveLeastSquares(A, b);
  }

  // Clone A and b to avoid mutating inputs
  const M = cloneMatrix(A);
  const y = b.slice();
  const P = new Array(n);
  for (let i = 0; i < n; i++) P[i] = i;

  // Forward elimination with partial pivoting
  for (let k = 0; k < n; k++) {
    // Find pivot row
    let maxVal = Math.abs(M[k][k]);
    let pivotRow = k;
    for (let i = k + 1; i < n; i++) {
      const absVal = Math.abs(M[i][k]);
      if (absVal > maxVal) {
        maxVal = absVal;
        pivotRow = i;
      }
    }

    if (maxVal < 1e-15) {
      return {
        x: createVector(n, 0),
        method: 'direct',
        iterations: 1,
        timeMs: performance.now() - startTime,
        residual: b.slice(),
        residualNorm: vectorNorm(b),
        relativeResidual: 1,
        converged: false,
        error: `Matrix is singular or near-singular (pivot element at row ${k} is ${maxVal.toExponential(2)}).`
      };
    }

    // Swap rows in M and y
    if (pivotRow !== k) {
      const tempRow = M[k];
      M[k] = M[pivotRow];
      M[pivotRow] = tempRow;

      const tempY = y[k];
      y[k] = y[pivotRow];
      y[pivotRow] = tempY;

      const tempP = P[k];
      P[k] = P[pivotRow];
      P[pivotRow] = tempP;
    }

    // Eliminate below pivot
    const pivot = M[k][k];
    for (let i = k + 1; i < n; i++) {
      const factor = M[i][k] / pivot;
      M[i][k] = 0;
      for (let j = k + 1; j < n; j++) {
        M[i][j] -= factor * M[k][j];
      }
      y[i] -= factor * y[k];
    }
  }

  // Back substitution (Ux = y)
  const x = createVector(n, 0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }

  const timeMs = Math.max(0.01, performance.now() - startTime);
  const Ax = matrixVectorMultiply(A, x);
  const residual = vectorSubtract(b, Ax);
  const residualNorm = vectorNorm(residual);
  const bNorm = vectorNorm(b);
  const relativeResidual = bNorm > 1e-14 ? residualNorm / bNorm : residualNorm;

  return {
    x,
    method: 'direct',
    iterations: 1,
    timeMs,
    residual,
    residualNorm,
    relativeResidual,
    converged: residualNorm < 1e-3 || relativeResidual < 1e-4
  };
}

/**
 * Solves Ax = b using Least Squares (QR or Normal Equations A^T A x = A^T b)
 * Corresponding to np.linalg.lstsq(A, b, rcond=None)[0]
 */
export function solveLeastSquares(A: Matrix, b: Vector): SolverResult {
  const startTime = performance.now();
  const m = A.length;
  const n = A[0]?.length || 0;

  // Form ATA = A^T * A (n x n) and ATb = A^T * b (n x 1)
  const ATA = new Array(n);
  for (let i = 0; i < n; i++) {
    ATA[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += A[k][i] * A[k][j];
      }
      ATA[i][j] = sum;
    }
  }

  // Add small ridge regularization if underdetermined or ill-conditioned
  if (m < n) {
    for (let i = 0; i < n; i++) {
      ATA[i][i] += 1e-9;
    }
  }

  const ATb = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = 0; k < m; k++) {
      sum += A[k][i] * b[k];
    }
    ATb[i] = sum;
  }

  // Solve ATA * x = ATb using LU
  const luResult = solveDirectLU(ATA, ATb);
  const timeMs = Math.max(0.01, performance.now() - startTime);

  const Ax = matrixVectorMultiply(A, luResult.x);
  const residual = vectorSubtract(b, Ax);
  const residualNorm = vectorNorm(residual);
  const bNorm = vectorNorm(b);
  const relativeResidual = bNorm > 1e-14 ? residualNorm / bNorm : residualNorm;

  return {
    x: luResult.x,
    method: 'lstsq',
    iterations: 1,
    timeMs,
    residual,
    residualNorm,
    relativeResidual,
    converged: !luResult.error,
    error: luResult.error
  };
}

/**
 * Educational Iterative Solver: Jacobi Method
 * x_i^(k+1) = (b_i - sum_{j!=i} a_ij * x_j^(k)) / a_ii
 */
export function solveJacobi(A: Matrix, b: Vector, maxIter = 200, tol = 1e-6): SolverResult {
  const startTime = performance.now();
  const n = A.length;
  let x = createVector(n, 0);
  const history: IterationStep[] = [];

  const bNorm = vectorNorm(b);
  const initialResidual = vectorSubtract(b, matrixVectorMultiply(A, x));
  history.push({
    iteration: 0,
    residualNorm: vectorNorm(initialResidual),
    xNorm: 0
  });

  // Check diagonal elements
  for (let i = 0; i < n; i++) {
    if (Math.abs(A[i][i]) < 1e-12) {
      return {
        x,
        method: 'jacobi',
        iterations: 0,
        timeMs: performance.now() - startTime,
        residual: initialResidual,
        residualNorm: vectorNorm(initialResidual),
        relativeResidual: 1,
        converged: false,
        error: `Jacobi failed: Zero or near-zero diagonal element at A[${i}, ${i}].`,
        history
      };
    }
  }

  let converged = false;
  let iter = 0;

  for (iter = 1; iter <= maxIter; iter++) {
    const xNew = createVector(n, 0);
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      const row = A[i];
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          sum -= row[j] * x[j];
        }
      }
      xNew[i] = sum / row[i];
    }

    const Ax = matrixVectorMultiply(A, xNew);
    const residual = vectorSubtract(b, Ax);
    const resNorm = vectorNorm(residual);
    const xNorm = vectorNorm(xNew);

    history.push({
      iteration: iter,
      residualNorm: resNorm,
      xNorm
    });

    x = xNew;

    if (resNorm < tol || (bNorm > 0 && resNorm / bNorm < tol)) {
      converged = true;
      break;
    }

    if (resNorm > 1e12 || isNaN(resNorm)) {
      // Divergence detected
      return {
        x,
        method: 'jacobi',
        iterations: iter,
        timeMs: performance.now() - startTime,
        residual,
        residualNorm: resNorm,
        relativeResidual: resNorm / (bNorm || 1),
        converged: false,
        error: `Jacobi diverged! The coefficient matrix is not strictly diagonally dominant.`,
        history
      };
    }
  }

  const timeMs = Math.max(0.01, performance.now() - startTime);
  const AxFinal = matrixVectorMultiply(A, x);
  const residualFinal = vectorSubtract(b, AxFinal);
  const residualNormFinal = vectorNorm(residualFinal);

  return {
    x,
    method: 'jacobi',
    iterations: Math.min(iter, maxIter),
    timeMs,
    residual: residualFinal,
    residualNorm: residualNormFinal,
    relativeResidual: bNorm > 1e-14 ? residualNormFinal / bNorm : residualNormFinal,
    converged,
    history
  };
}

/**
 * Educational Iterative Solver: Gauss-Seidel Method
 * Uses updated values immediately in the current iteration
 */
export function solveGaussSeidel(A: Matrix, b: Vector, maxIter = 200, tol = 1e-6): SolverResult {
  const startTime = performance.now();
  const n = A.length;
  const x = createVector(n, 0);
  const history: IterationStep[] = [];

  const bNorm = vectorNorm(b);
  const initialResidual = vectorSubtract(b, matrixVectorMultiply(A, x));
  history.push({
    iteration: 0,
    residualNorm: vectorNorm(initialResidual),
    xNorm: 0
  });

  // Check diagonal elements
  for (let i = 0; i < n; i++) {
    if (Math.abs(A[i][i]) < 1e-12) {
      return {
        x,
        method: 'gauss-seidel',
        iterations: 0,
        timeMs: performance.now() - startTime,
        residual: initialResidual,
        residualNorm: vectorNorm(initialResidual),
        relativeResidual: 1,
        converged: false,
        error: `Gauss-Seidel failed: Zero or near-zero diagonal element at A[${i}, ${i}].`,
        history
      };
    }
  }

  let converged = false;
  let iter = 0;

  for (iter = 1; iter <= maxIter; iter++) {
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      const row = A[i];
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          sum -= row[j] * x[j];
        }
      }
      x[i] = sum / row[i];
    }

    const Ax = matrixVectorMultiply(A, x);
    const residual = vectorSubtract(b, Ax);
    const resNorm = vectorNorm(residual);
    const xNorm = vectorNorm(x);

    history.push({
      iteration: iter,
      residualNorm: resNorm,
      xNorm
    });

    if (resNorm < tol || (bNorm > 0 && resNorm / bNorm < tol)) {
      converged = true;
      break;
    }

    if (resNorm > 1e12 || isNaN(resNorm)) {
      // Divergence detected
      return {
        x,
        method: 'gauss-seidel',
        iterations: iter,
        timeMs: performance.now() - startTime,
        residual,
        residualNorm: resNorm,
        relativeResidual: resNorm / (bNorm || 1),
        converged: false,
        error: `Gauss-Seidel diverged! System is not diagonally dominant or symmetric positive definite.`,
        history
      };
    }
  }

  const timeMs = Math.max(0.01, performance.now() - startTime);
  const AxFinal = matrixVectorMultiply(A, x);
  const residualFinal = vectorSubtract(b, AxFinal);
  const residualNormFinal = vectorNorm(residualFinal);

  return {
    x,
    method: 'gauss-seidel',
    iterations: Math.min(iter, maxIter),
    timeMs,
    residual: residualFinal,
    residualNorm: residualNormFinal,
    relativeResidual: bNorm > 1e-14 ? residualNormFinal / bNorm : residualNormFinal,
    converged,
    history
  };
}
