import React, { useMemo, useState } from 'react';
import type { Matrix, Vector, SolverType } from '../../math/types';
import { Copy, Download, Code2, Check } from 'lucide-react';

interface PythonLabProps {
  A: Matrix;
  b: Vector;
  solverMethod: SolverType;
}

export const PythonLab: React.FC<PythonLabProps> = ({ A, b, solverMethod }) => {
  const rows = A.length;
  const cols = A[0]?.length || 0;
  const isLarge = rows > 8 || cols > 8;
  const [copied, setCopied] = useState(false);

  const code = useMemo(() => {
    let matrixStr: string;
    if (isLarge) {
      matrixStr = `# Matrix A is ${rows}×${cols} — generated programmatically\nimport numpy as np\nfrom numpy.random import default_rng\n\nrng = default_rng(seed=42)\nA = rng.standard_normal((${rows}, ${cols}))`;
    } else {
      const rowStrs = A.map(row => '    [' + row.map(v => Number.isInteger(v) ? v.toString() : v.toFixed(4)).join(', ') + ']');
      matrixStr = `import numpy as np\n\nA = np.array([\n${rowStrs.join(',\n')}\n])`;
    }

    let bStr: string;
    if (isLarge) {
      bStr = `b = rng.standard_normal(${rows})`;
    } else {
      bStr = `b = np.array([${b.map(v => Number.isInteger(v) ? v.toString() : v.toFixed(4)).join(', ')}])`;
    }

    let solveStr: string;
    switch (solverMethod) {
      case 'lstsq':
        solveStr = `# Least Squares Solution (np.linalg.lstsq)\nx, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)\nprint("Solution vector x:", x)`;
        break;
      case 'jacobi':
        solveStr = `# Jacobi Iterative Solver (Diagonal Splitting)\ndef jacobi(A, b, tol=1e-6, max_iter=200):\n    n = len(b)\n    x = np.zeros(n)\n    D = np.diag(A)\n    R = A - np.diagflat(D)\n    for k in range(max_iter):\n        x_new = (b - R @ x) / D\n        if np.linalg.norm(x_new - x) < tol:\n            print(f"Converged in {k+1} iterations")\n            return x_new\n        x = x_new\n    print("Max iterations reached")\n    return x\n\nx = jacobi(A, b)\nprint("Solution vector x:", x)`;
        break;
      case 'gauss-seidel':
        solveStr = `# Gauss-Seidel Iterative Solver\ndef gauss_seidel(A, b, tol=1e-6, max_iter=200):\n    n = len(b)\n    x = np.zeros(n)\n    for k in range(max_iter):\n        x_old = x.copy()\n        for i in range(n):\n            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)\n            x[i] = (b[i] - sigma) / A[i][i]\n        if np.linalg.norm(x - x_old) < tol:\n            print(f"Converged in {k+1} iterations")\n            return x\n    print("Max iterations reached")\n    return x\n\nx = gauss_seidel(A, b)\nprint("Solution vector x:", x)`;
        break;
      default:
        solveStr = `# Direct LU Decomposition with Partial Pivoting\nx = np.linalg.solve(A, b)\nprint("Solution vector x:", x)`;
    }

    const verifyStr = `\n# ─── Residual & Condition Verification ───\nresidual = b - A @ x\nresidual_norm = np.linalg.norm(residual)\nprint(f"Residual ‖Ax - b‖₂: {residual_norm:.4e}")\n\nkappa = np.linalg.cond(A)\nprint(f"Condition number κ(A): {kappa:.4e}")`;

    return `${matrixStr}\n\n${bStr}\n\nprint(f"Linear System: {A.shape[0]}×{A.shape[1]}")\n${solveStr}\n${verifyStr}`;
  }, [A, b, rows, cols, isLarge, solverMethod]);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'solve_linear_system.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full flex flex-col gap-4">
      {/* Top Header Card */}
      <div className="flex items-center justify-between px-5 py-3.5 rounded-2xl bg-white border border-[#E6E9EE] shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-[#EBF7EF] border border-[#99D4AA] flex items-center justify-center text-[#276239]">
            <Code2 size={18} />
          </div>
          <div>
            <h4 className="text-sm font-bold text-[#2D3136]">Python NumPy Script</h4>
            <p className="text-xs text-[#7E8694]">Exact executable script reproducing your active system</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="btn btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
          >
            {copied ? <Check size={14} className="text-[#68BA7F]" /> : <Copy size={14} />}
            {copied ? 'Copied!' : 'Copy Code'}
          </button>
          <button
            onClick={handleDownload}
            className="btn btn-primary text-xs py-1.5 px-3 flex items-center gap-1.5"
          >
            <Download size={14} /> Download .py
          </button>
        </div>
      </div>

      {/* Code Window Container */}
      <div className="relative rounded-2xl border border-[#CBCBCB] bg-[#22252A] overflow-hidden shadow-md">
        {/* Terminal Header Bar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[#1A1C20] border-b border-white/10">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#FFA896]" />
            <div className="w-3 h-3 rounded-full bg-[#FFF0C8]" />
            <div className="w-3 h-3 rounded-full bg-[#CFFFDC]" />
            <span className="ml-2 text-xs text-[#A4ACB9] font-mono font-medium">solve_linear_system.py</span>
          </div>
          <span className="text-[11px] font-mono text-[#7E8694]">Python 3 · NumPy · LAPACK</span>
        </div>

        {/* Code Content */}
        <pre className="p-5 overflow-auto max-h-[520px] text-[13px] leading-[1.8] text-[#E2E8F0] font-mono whitespace-pre">
          <code>{highlightPython(code)}</code>
        </pre>
      </div>
    </div>
  );
};

function highlightPython(code: string): React.ReactNode {
  const lines = code.split('\n');
  return lines.map((line, i) => {
    if (line.trimStart().startsWith('#')) {
      return (
        <span key={i} className="text-[#8E97A4] italic">
          {line}{'\n'}
        </span>
      );
    }

    const parts: React.ReactNode[] = [];
    const keywords = /\b(import|from|as|def|for|in|range|if|return|print|sum|len|None|True|False)\b/g;
    let lastIndex = 0;
    let match;

    while ((match = keywords.exec(line)) !== null) {
      if (match.index > lastIndex) {
        parts.push(<span key={`${i}-${lastIndex}`}>{line.slice(lastIndex, match.index)}</span>);
      }
      parts.push(
        <span key={`${i}-${match.index}`} className="text-[#CFFFDC] font-semibold">
          {match[0]}
        </span>
      );
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < line.length) {
      parts.push(<span key={`${i}-rest`}>{line.slice(lastIndex)}</span>);
    }

    return (
      <span key={i}>
        {parts}
        {'\n'}
      </span>
    );
  });
}
