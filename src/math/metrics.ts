import type { Matrix, MatrixMetrics } from './types';
import { cloneMatrix, matrixNormL1, matrixNormFrobenius } from './matrix';
import { solveDirectLU } from './solvers';

/**
 * Computes matrix metrics including Rank, Norms, Condition Number, Sparsity, Symmetry, and Positive Definiteness
 */
export function computeMatrixMetrics(A: Matrix): MatrixMetrics {
  const rows = A.length;
  const cols = A[0]?.length || 0;
  const totalCount = rows * cols;

  let nonZeroCount = 0;
  let min = Infinity;
  let max = -Infinity;
  let isSymmetric = rows === cols;

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const val = A[i][j];
      if (Math.abs(val) > 1e-12) nonZeroCount++;
      if (val < min) min = val;
      if (val > max) max = val;

      if (isSymmetric && j > i) {
        if (Math.abs(val - A[j][i]) > 1e-6) {
          isSymmetric = false;
        }
      }
    }
  }

  const sparsity = totalCount > 0 ? ((totalCount - nonZeroCount) / totalCount) * 100 : 0;
  const frobeniusNorm = matrixNormFrobenius(A);
  const l1Norm = matrixNormL1(A);

  // Determinant & Rank via LU decomposition (only for N <= 100 to stay fast)
  let determinant: number | null = null;
  let rank = Math.min(rows, cols);
  let isPositiveDefinite = false;

  if (rows === cols && rows <= 150) {
    const M = cloneMatrix(A);
    let det = 1;
    let rankCount = 0;
    let isSingular = false;

    for (let k = 0; k < rows; k++) {
      let maxVal = Math.abs(M[k][k]);
      let pivotRow = k;
      for (let i = k + 1; i < rows; i++) {
        const absVal = Math.abs(M[i][k]);
        if (absVal > maxVal) {
          maxVal = absVal;
          pivotRow = i;
        }
      }

      if (maxVal < 1e-12) {
        isSingular = true;
        continue;
      }

      rankCount++;

      if (pivotRow !== k) {
        const temp = M[k];
        M[k] = M[pivotRow];
        M[pivotRow] = temp;
        det = -det;
      }

      const pivot = M[k][k];
      det *= pivot;

      for (let i = k + 1; i < rows; i++) {
        const factor = M[i][k] / pivot;
        for (let j = k + 1; j < rows; j++) {
          M[i][j] -= factor * M[k][j];
        }
      }
    }

    rank = isSingular ? rankCount : rows;
    determinant = isSingular ? 0 : det;

    // Check positive definiteness via Cholesky decomposition if symmetric
    if (isSymmetric) {
      isPositiveDefinite = true;
      const L = cloneMatrix(A);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j <= i; j++) {
          let sum = 0;
          for (let k = 0; k < j; k++) {
            sum += L[i][k] * L[j][k];
          }
          if (i === j) {
            const val = L[i][i] - sum;
            if (val <= 1e-12) {
              isPositiveDefinite = false;
              break;
            }
            L[i][i] = Math.sqrt(val);
          } else {
            L[i][j] = (L[i][j] - sum) / L[j][j];
          }
        }
        if (!isPositiveDefinite) break;
      }
    }
  }

  // Condition number estimation: kappa(A) = ||A|| * ||A^-1||
  // Uses Hager's 1-norm estimation algorithm via solving Ax = y with test vectors
  let conditionNumber = 1.0;
  if (rows === cols && rows > 0) {
    if (rows === 1) {
      conditionNumber = 1.0;
    } else if (rows <= 100) {
      // Hager-Higham 1-norm condition estimator
      try {
        let invNormEst = 0;
        // Test standard basis probe vectors
        const probeCount = Math.min(5, rows);
        for (let p = 0; p < probeCount; p++) {
          const probe = new Array(rows).fill(0);
          probe[p] = 1;
          const sol = solveDirectLU(A, probe);
          if (sol.converged) {
            let solNorm = 0;
            for (let i = 0; i < rows; i++) solNorm += Math.abs(sol.x[i]);
            if (solNorm > invNormEst) invNormEst = solNorm;
          }
        }
        conditionNumber = invNormEst > 0 ? l1Norm * invNormEst : 1e16;
      } catch {
        conditionNumber = 1e16;
      }
    } else {
      // Fast heuristic estimate for very large matrices
      const diagMin = Math.min(...A.map((row, i) => Math.abs(row[i] || 1)));
      conditionNumber = diagMin > 1e-12 ? (l1Norm / diagMin) : 1e16;
    }
  }

  return {
    rows,
    cols,
    rank,
    frobeniusNorm,
    l2Norm: frobeniusNorm / Math.sqrt(Math.min(rows, cols)), // Fast bound
    conditionNumber: isFinite(conditionNumber) && conditionNumber >= 1 ? conditionNumber : 1e16,
    sparsity: Math.round(sparsity * 100) / 100,
    nonZeroCount,
    totalCount,
    isSymmetric,
    isPositiveDefinite,
    determinant: determinant !== null && isFinite(determinant) ? determinant : null,
    min: isFinite(min) ? min : 0,
    max: isFinite(max) ? max : 0
  };
}
