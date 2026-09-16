import type { Matrix, Vector, SystemConfig } from './types';
import { PRNG } from './prng';
import { createMatrix, createVector } from './matrix';

export interface GeneratedSystem {
  A: Matrix;
  b: Vector;
  description: string;
  conditionNumberEstimate?: number;
}

/**
 * Generates matrix A and vector b based on SystemConfig presets
 */
export function generateSystem(config: SystemConfig): GeneratedSystem {
  const prng = new PRNG(config.seed);
  const n = config.size;
  let rows = config.rows || n;
  let cols = config.cols || n;

  if (config.preset === 'overdetermined') {
    rows = Math.max(3, n);
    cols = Math.max(2, Math.floor(n * 0.6));
  } else if (config.preset === 'underdetermined') {
    cols = Math.max(3, n);
    rows = Math.max(2, Math.floor(n * 0.6));
  }

  let A: Matrix = createMatrix(rows, cols, 0);
  let description = '';

  switch (config.preset) {
    case 'identity': {
      description = `Identity Matrix (${rows}×${cols}): Diagonal elements are 1, off-diagonal are 0. Condition number κ(A) = 1.`;
      for (let i = 0; i < Math.min(rows, cols); i++) {
        A[i][i] = 1;
      }
      break;
    }

    case 'diagonal': {
      description = `Diagonal Matrix: Non-zero values exist only on main diagonal. Decoupled linear equations.`;
      for (let i = 0; i < Math.min(rows, cols); i++) {
        const val = config.distribution === 'normal' ? prng.normal(5, 2) : prng.uniform(1, 10);
        A[i][i] = Math.round(val * 100) / 100;
      }
      break;
    }

    case 'symmetric': {
      description = `Symmetric Matrix (A = Aᵀ): Satisfies a_ij = a_ji. Important in physics, mechanics, and optimization.`;
      const dim = Math.min(rows, cols);
      rows = dim;
      cols = dim;
      A = createMatrix(dim, dim, 0);
      for (let i = 0; i < dim; i++) {
        for (let j = i; j < dim; j++) {
          const val = config.distribution === 'normal' ? prng.normal(0, 5) : prng.uniform(-10, 10);
          const rounded = Math.round(val * 100) / 100;
          A[i][j] = rounded;
          A[j][i] = rounded;
        }
      }
      break;
    }

    case 'positive-definite': {
      description = `Symmetric Positive Definite (SPD): xᵀAx > 0 for all x ≠ 0. Guaranteed unique solution and stable Cholesky factorization.`;
      const dim = Math.min(rows, cols);
      rows = dim;
      cols = dim;
      A = createMatrix(dim, dim, 0);
      // Construct A = M^T * M + alpha * I
      const M = createMatrix(dim, dim, 0);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          M[i][j] = prng.uniform(-2, 2);
        }
      }
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          let dot = 0;
          for (let k = 0; k < dim; k++) {
            dot += M[k][i] * M[k][j];
          }
          A[i][j] = Math.round(dot * 100) / 100;
        }
        A[i][i] += Math.round((dim * 0.5 + 2) * 100) / 100; // Strict diagonal dominance guarantees SPD
      }
      break;
    }

    case 'tridiagonal': {
      description = `Tridiagonal Matrix: Non-zero values only on main diagonal and immediately adjacent sub/super-diagonals. Common in finite difference PDE discretizations.`;
      for (let i = 0; i < Math.min(rows, cols); i++) {
        A[i][i] = Math.round(prng.uniform(4, 8) * 100) / 100;
        if (i > 0) {
          const sub = Math.round(prng.uniform(-2, -0.5) * 100) / 100;
          A[i][i - 1] = sub;
          A[i - 1][i] = sub;
        }
      }
      break;
    }

    case 'banded': {
      const bw = Math.max(1, config.bandwidth || 2);
      description = `Banded Matrix (Bandwidth = ${bw}): Non-zero elements confined to a band around the main diagonal.`;
      for (let i = 0; i < rows; i++) {
        for (let j = Math.max(0, i - bw); j <= Math.min(cols - 1, i + bw); j++) {
          const val = i === j ? prng.uniform(5, 10) : prng.uniform(-2, 2);
          A[i][j] = Math.round(val * 100) / 100;
        }
      }
      break;
    }

    case 'sparse': {
      const density = 1 - (config.sparsityFactor || 0.85);
      description = `Sparse Matrix: Only ~${Math.round(density * 100)}% non-zero elements. Storing and solving requires specialized sparse storage.`;
      for (let i = 0; i < rows; i++) {
        // Ensure at least diagonal is non-zero so matrix is invertible
        A[i][i % cols] = Math.round(prng.uniform(3, 8) * 100) / 100;
        for (let j = 0; j < cols; j++) {
          if (i !== j && prng.nextFloat() < density) {
            const val = prng.uniform(-5, 5);
            A[i][j] = Math.round(val * 100) / 100;
          }
        }
      }
      break;
    }

    case 'ill-conditioned': {
      description = `Ill-Conditioned System (Hilbert / Vandermonde): Extremely high condition number κ(A) >> 10⁶. Tiny perturbations in b cause massive errors in solution x.`;
      const dim = Math.min(rows, cols);
      rows = dim;
      cols = dim;
      A = createMatrix(dim, dim, 0);
      // Hilbert matrix H_ij = 1 / (i + j + 1)
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          A[i][j] = 1 / (i + j + 1);
        }
      }
      break;
    }

    case 'overdetermined': {
      description = `Overdetermined System (${rows} equations > ${cols} unknowns): More equations than variables. Solved via Least Squares (np.linalg.lstsq).`;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const val = prng.uniform(-4, 4);
          A[i][j] = Math.round(val * 100) / 100;
        }
      }
      break;
    }

    case 'underdetermined': {
      description = `Underdetermined System (${rows} equations < ${cols} unknowns): Fewer equations than variables. Infinitely many solutions; Least Squares finds minimum-norm solution.`;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const val = prng.uniform(-4, 4);
          A[i][j] = Math.round(val * 100) / 100;
        }
      }
      break;
    }

    case 'random':
    default: {
      description = `Random Dense Linear System (${rows}×${cols}): Generated with seed ${config.seed} (${config.distribution} distribution).`;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const val = config.distribution === 'normal' ? prng.normal(0, 3) : prng.uniform(-5, 5);
          A[i][j] = Math.round(val * 100) / 100;
        }
        // Add diagonal dominance boost to avoid random singularity
        if (i < cols) {
          A[i][i] += Math.sign(A[i][i] || 1) * (Math.abs(A[i][i]) + Math.sqrt(cols) * 1.5);
          A[i][i] = Math.round(A[i][i] * 100) / 100;
        }
      }
      break;
    }
  }

  // Generate vector b
  const b: Vector = createVector(rows, 0);
  if (config.preset === 'ill-conditioned') {
    // Exact b such that exact solution is x = [1, 1, ..., 1] for easy demonstration
    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        rowSum += A[i][j];
      }
      b[i] = rowSum;
    }
  } else {
    for (let i = 0; i < rows; i++) {
      const val = config.distribution === 'normal' ? prng.normal(1, 4) : prng.uniform(-10, 10);
      b[i] = Math.round(val * 100) / 100;
    }
  }

  return {
    A,
    b,
    description
  };
}
