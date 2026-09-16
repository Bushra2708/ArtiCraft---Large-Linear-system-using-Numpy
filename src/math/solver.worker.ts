import type { Matrix, Vector, SolverType } from './types';
import { solveDirectLU, solveLeastSquares, solveJacobi, solveGaussSeidel } from './solvers';

export interface WorkerSolveRequest {
  type: 'solve' | 'benchmark';
  A: Matrix;
  b: Vector;
  method?: SolverType;
  maxIter?: number;
  tol?: number;
}

self.onmessage = (e: MessageEvent<WorkerSolveRequest>) => {
  const { type, A, b, method, maxIter, tol } = e.data;

  if (type === 'solve') {
    let result;
    switch (method) {
      case 'lstsq':
        result = solveLeastSquares(A, b);
        break;
      case 'jacobi':
        result = solveJacobi(A, b, maxIter, tol);
        break;
      case 'gauss-seidel':
        result = solveGaussSeidel(A, b, maxIter, tol);
        break;
      case 'direct':
      default:
        result = solveDirectLU(A, b);
        break;
    }
    self.postMessage({ type: 'solve-result', result });
  } else if (type === 'benchmark') {
    // Run all solvers and return comparative benchmark
    const directRes = solveDirectLU(A, b);
    const lstsqRes = solveLeastSquares(A, b);
    const jacobiRes = solveJacobi(A, b, maxIter || 100, tol || 1e-5);
    const gsRes = solveGaussSeidel(A, b, maxIter || 100, tol || 1e-5);

    self.postMessage({
      type: 'benchmark-result',
      benchmark: [
        { name: 'NumPy Direct Solve (LU)', result: directRes, note: 'Standard np.linalg.solve' },
        { name: 'Gauss-Seidel Iterative', result: gsRes, note: 'Successive updates' },
        { name: 'Jacobi Iterative', result: jacobiRes, note: 'Simultaneous updates' },
        { name: 'Least Squares (SVD/Normal)', result: lstsqRes, note: 'np.linalg.lstsq' }
      ]
    });
  }
};
