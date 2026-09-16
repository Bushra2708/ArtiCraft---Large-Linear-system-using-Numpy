export type Matrix = number[][];
export type Vector = number[];

export type SolverType = 'direct' | 'lstsq' | 'jacobi' | 'gauss-seidel';

export type MatrixPreset = 
  | 'random'
  | 'identity'
  | 'diagonal'
  | 'symmetric'
  | 'positive-definite'
  | 'sparse'
  | 'banded'
  | 'tridiagonal'
  | 'ill-conditioned'
  | 'overdetermined'
  | 'underdetermined'
  | 'manual'
  | 'python';

export interface IterationStep {
  iteration: number;
  residualNorm: number;
  xNorm: number;
}

export interface SolverResult {
  x: Vector;
  method: SolverType;
  iterations: number;
  timeMs: number;
  residual: Vector;
  residualNorm: number;
  relativeResidual: number;
  converged: boolean;
  history?: IterationStep[];
  error?: string;
  isApproximate?: boolean;
}

export interface MatrixMetrics {
  rows: number;
  cols: number;
  rank: number;
  frobeniusNorm: number;
  l2Norm: number;
  conditionNumber: number;
  sparsity: number; // percentage, e.g. 75.4
  nonZeroCount: number;
  totalCount: number;
  isSymmetric: boolean;
  isPositiveDefinite: boolean;
  determinant: number | null;
  min: number;
  max: number;
}

export interface SystemConfig {
  size: number;
  rows: number;
  cols: number;
  preset: MatrixPreset;
  distribution: 'uniform' | 'normal';
  seed: number;
  sparsityFactor: number; // 0 to 1
  noiseLevel: number;
  bandwidth: number;
}

export interface RowColStats {
  index: number;
  type: 'row' | 'col';
  values: number[];
  nonZeroCount: number;
  min: number;
  max: number;
  mean: number;
  norm: number;
}
