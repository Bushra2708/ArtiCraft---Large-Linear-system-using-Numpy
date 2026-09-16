import type { Matrix, Vector, RowColStats } from './types';

/**
 * Creates an empty or pre-filled m x n matrix
 */
export function createMatrix(rows: number, cols: number, initialValue: number = 0): Matrix {
  const mat: Matrix = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols);
    for (let j = 0; j < cols; j++) {
      row[j] = initialValue;
    }
    mat[i] = row;
  }
  return mat;
}

/**
 * Creates an empty or pre-filled vector of length n
 */
export function createVector(length: number, initialValue: number = 0): Vector {
  const vec = new Array(length);
  for (let i = 0; i < length; i++) {
    vec[i] = initialValue;
  }
  return vec;
}

/**
 * Deep clones a matrix
 */
export function cloneMatrix(A: Matrix): Matrix {
  const rows = A.length;
  const copy: Matrix = new Array(rows);
  for (let i = 0; i < rows; i++) {
    copy[i] = A[i].slice();
  }
  return copy;
}

/**
 * Multiplies Matrix A (m x n) by Vector x (n) to produce Vector Ax (m)
 */
export function matrixVectorMultiply(A: Matrix, x: Vector): Vector {
  const m = A.length;
  const n = A[0]?.length || 0;
  const result = new Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    const row = A[i];
    for (let j = 0; j < n; j++) {
      sum += row[j] * x[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * Vector subtraction: a - b
 */
export function vectorSubtract(a: Vector, b: Vector): Vector {
  const n = Math.min(a.length, b.length);
  const diff = new Array(n);
  for (let i = 0; i < n; i++) {
    diff[i] = a[i] - b[i];
  }
  return diff;
}

/**
 * Vector addition: a + b
 */
export function vectorAdd(a: Vector, b: Vector): Vector {
  const n = Math.min(a.length, b.length);
  const sum = new Array(n);
  for (let i = 0; i < n; i++) {
    sum[i] = a[i] + b[i];
  }
  return sum;
}

/**
 * Scalar multiplication: c * v
 */
export function vectorScale(v: Vector, c: number): Vector {
  return v.map(val => val * c);
}

/**
 * Euclidean norm ||v||_2
 */
export function vectorNorm(v: Vector): number {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) {
    sumSq += v[i] * v[i];
  }
  return Math.sqrt(sumSq);
}

/**
 * Vector dot product
 */
export function dotProduct(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Frobenius norm of matrix A: sqrt(sum(A_ij^2))
 */
export function matrixNormFrobenius(A: Matrix): number {
  let sum = 0;
  const rows = A.length;
  const cols = A[0]?.length || 0;
  for (let i = 0; i < rows; i++) {
    const row = A[i];
    for (let j = 0; j < cols; j++) {
      sum += row[j] * row[j];
    }
  }
  return Math.sqrt(sum);
}

/**
 * L1 matrix norm: max column sum of absolute values
 */
export function matrixNormL1(A: Matrix): number {
  const rows = A.length;
  const cols = A[0]?.length || 0;
  let maxColSum = 0;
  for (let j = 0; j < cols; j++) {
    let colSum = 0;
    for (let i = 0; i < rows; i++) {
      colSum += Math.abs(A[i][j]);
    }
    if (colSum > maxColSum) maxColSum = colSum;
  }
  return maxColSum;
}

/**
 * Infinity matrix norm: max row sum of absolute values
 */
export function matrixNormInf(A: Matrix): number {
  const rows = A.length;
  const cols = A[0]?.length || 0;
  let maxRowSum = 0;
  for (let i = 0; i < rows; i++) {
    let rowSum = 0;
    for (let j = 0; j < cols; j++) {
      rowSum += Math.abs(A[i][j]);
    }
    if (rowSum > maxRowSum) maxRowSum = rowSum;
  }
  return maxRowSum;
}

/**
 * Matrix transpose: A^T
 */
export function transpose(A: Matrix): Matrix {
  const rows = A.length;
  const cols = A[0]?.length || 0;
  const AT = createMatrix(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      AT[j][i] = A[i][j];
    }
  }
  return AT;
}

/**
 * Computes Row or Column statistics for inspection
 */
export function computeRowColStats(A: Matrix, type: 'row' | 'col', index: number): RowColStats {
  const rows = A.length;
  const cols = A[0]?.length || 0;
  const values: number[] = [];

  if (type === 'row') {
    const r = Math.max(0, Math.min(index, rows - 1));
    for (let j = 0; j < cols; j++) {
      values.push(A[r][j]);
    }
  } else {
    const c = Math.max(0, Math.min(index, cols - 1));
    for (let i = 0; i < rows; i++) {
      values.push(A[i][c]);
    }
  }

  let nonZeroCount = 0;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;

  for (const v of values) {
    if (Math.abs(v) > 1e-12) nonZeroCount++;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
  }

  return {
    index,
    type,
    values,
    nonZeroCount,
    min: values.length > 0 ? min : 0,
    max: values.length > 0 ? max : 0,
    mean: values.length > 0 ? sum / values.length : 0,
    norm: Math.sqrt(sumSq)
  };
}
