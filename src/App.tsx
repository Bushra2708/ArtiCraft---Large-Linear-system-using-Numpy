import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  Play, ChevronLeft, ChevronRight, Shuffle, Layers, Code2,
  BarChart3, Activity, CheckCircle2, AlertTriangle, Search,
  X, ArrowDown, GitCompare,
  Eye, Grid3X3, BookOpen, Cpu, Sun, Moon,
  Sliders, ArrowUpRight, Sparkles, RefreshCw, ChevronDown, Check,
  PanelLeftClose, PanelLeftOpen, Menu
} from 'lucide-react';
import type { Matrix, Vector, SolverResult, SolverType, MatrixPreset, SystemConfig, MatrixMetrics, RowColStats } from './math/types';
import { generateSystem } from './math/generators';
import { solveDirectLU, solveLeastSquares, solveJacobi, solveGaussSeidel } from './math/solvers';
import { computeMatrixMetrics } from './math/metrics';
import { matrixVectorMultiply, vectorNorm, computeRowColStats } from './math/matrix';
import { parseMatrixText, parseVectorText, validateSystem } from './math/parser';
import { MatrixHeatmapCanvas } from './components/matrix/MatrixHeatmapCanvas';
import { SparsityCanvas } from './components/matrix/SparsityCanvas';
import { VectorView, ConvergenceChart, VerificationView } from './components/visualization/Visualizations';
import { PythonLab } from './components/python/PythonLab';

type AppTab = 'system' | 'matrix' | 'solver' | 'analysis' | 'python' | 'compare';
type MatrixViewMode = 'heatmap' | 'sparsity' | 'values';
type SystemStatus = 'ready' | 'generating' | 'solving' | 'solved' | 'error' | 'singular' | 'warning';

const PRESETS: { key: MatrixPreset; label: string; icon: React.ReactNode; desc: string; category: string }[] = [
  { key: 'random',            label: 'Random Dense',       icon: <Shuffle size={13} />,       desc: 'Full dense Gaussian matrix', category: 'Standard' },
  { key: 'identity',          label: 'Identity',           icon: <Grid3X3 size={13} />,        desc: 'I·x = b, perfectly stable (κ=1)', category: 'Standard' },
  { key: 'diagonal',          label: 'Diagonal',           icon: <Layers size={13} />,         desc: 'Decoupled diagonal equations', category: 'Standard' },
  { key: 'symmetric',         label: 'Symmetric',          icon: <GitCompare size={13} />,     desc: 'A = Aᵀ symmetric matrix', category: 'Structured' },
  { key: 'positive-definite', label: 'Pos. Definite',      icon: <CheckCircle2 size={13} />,   desc: 'xᵀAx > 0 strictly positive eigenvalues', category: 'Structured' },
  { key: 'tridiagonal',       label: 'Tridiagonal',        icon: <Activity size={13} />,       desc: 'Bandwidth = 1 (finite difference style)', category: 'Sparse' },
  { key: 'banded',            label: 'Banded',             icon: <BarChart3 size={13} />,      desc: 'Narrow bandwidth diagonal bands', category: 'Sparse' },
  { key: 'sparse',            label: 'Sparse',             icon: <Sparkles size={13} />,       desc: 'High zero-percentage structure', category: 'Sparse' },
  { key: 'ill-conditioned',   label: 'Ill-Conditioned',    icon: <AlertTriangle size={13} />,  desc: 'Hilbert matrix with explosive condition κ', category: 'Stress Test' },
  { key: 'overdetermined',    label: 'Overdetermined',     icon: <ArrowDown size={13} />,      desc: 'm > n system solved via Least Squares', category: 'Rectangular' },
  { key: 'underdetermined',   label: 'Underdetermined',    icon: <ArrowDown style={{ transform: 'rotate(180deg)' }} size={13} />, desc: 'm < n minimum-norm solution', category: 'Rectangular' },
];

const SOLVER_OPTIONS: { key: SolverType; label: string; desc: string; tag: string }[] = [
  { key: 'direct',       label: 'NumPy Direct (LU)',     desc: 'Gaussian elimination w/ partial pivoting — np.linalg.solve()', tag: 'Recommended' },
  { key: 'lstsq',        label: 'Least Squares (lstsq)', desc: 'Normal equations via SVD/QR — np.linalg.lstsq()',            tag: 'Overdetermined' },
  { key: 'jacobi',       label: 'Jacobi Iterative',      desc: 'Diagonal splitting iteration for dominant systems',           tag: 'Iterative' },
  { key: 'gauss-seidel', label: 'Gauss-Seidel',          desc: 'Successive displacement — faster iterative convergence',       tag: 'Iterative' },
];

const DEFAULT_CONFIG: SystemConfig = {
  size: 5, rows: 5, cols: 5, preset: 'random',
  distribution: 'uniform', seed: 42, sparsityFactor: 0.85, noiseLevel: 0.01, bandwidth: 2
};

const fmt = (v: number, digits = 4) => {
  if (!isFinite(v)) return '∞';
  if (Math.abs(v) === 0) return '0';
  if (Math.abs(v) < 0.001 || Math.abs(v) > 99999) return v.toExponential(digits > 2 ? 2 : digits);
  return v.toFixed(digits);
};

export default function App() {
  // ─── State ───
  const [config, setConfig] = useState<SystemConfig>(DEFAULT_CONFIG);
  const [A, setA] = useState<Matrix>([]);
  const [b, setB] = useState<Vector>([]);
  const [metrics, setMetrics] = useState<MatrixMetrics | null>(null);
  const [solverMethod, setSolverMethod] = useState<SolverType>('direct');
  const [solverResult, setSolverResult] = useState<SolverResult | null>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<{ name: string; result: SolverResult }[]>([]);
  const [status, setStatus] = useState<SystemStatus>('ready');
  const [activeTab, setActiveTab] = useState<AppTab>('system');
  const [matrixView, setMatrixView] = useState<MatrixViewMode>('heatmap');
  const [commandOpen, setCommandOpen] = useState(false);
  const [commandFilter, setCommandFilter] = useState('');
  const [commandSelectedIndex, setCommandSelectedIndex] = useState(0);
  const [presentationMode, setPresentationMode] = useState(false);
  const [presentationStep, setPresentationStep] = useState(0);
  const [inputMode, setInputMode] = useState<'generate' | 'manual' | 'python'>('generate');
  const [manualA, setManualA] = useState('3 2 -1\n2 -2 4\n-1 0.5 -1');
  const [manualB, setManualB] = useState('1\n-2\n0');
  const [pythonInput, setPythonInput] = useState('A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]])\nb = np.array([1, -2, 0])');
  const [rowColStats, setRowColStats] = useState<RowColStats | null>(null);
  const [solveAnimation, setSolveAnimation] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [themeMode, setThemeMode] = useState<'light' | 'dark'>('light');
  const [showConfigDrawer, setShowConfigDrawer] = useState(false);
  const [presetDropdownOpen, setPresetDropdownOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [systemConfigOpen, setSystemConfigOpen] = useState(true);

  const cmdInputRef = useRef<HTMLInputElement>(null);
  const presetDropdownRef = useRef<HTMLDivElement>(null);

  // Sync theme attribute to html document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', themeMode);
  }, [themeMode]);

  // Click outside preset dropdown
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (presetDropdownRef.current && !presetDropdownRef.current.contains(e.target as Node)) {
        setPresetDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Keyboard shortcuts (Ctrl+K, Esc)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setCommandOpen(o => !o);
        setCommandFilter('');
        setCommandSelectedIndex(0);
      }
      if (e.key === 'Escape') {
        setCommandOpen(false);
        setPresentationMode(false);
        setShowConfigDrawer(false);
        setPresetDropdownOpen(false);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  useEffect(() => {
    if (commandOpen && cmdInputRef.current) cmdInputRef.current.focus();
  }, [commandOpen]);

  // ─── Generate System ───
  const handleGenerate = useCallback((configOverride?: SystemConfig) => {
    setStatus('generating');
    setErrorMsg(null);
    setSolverResult(null);
    setBenchmarkResults([]);
    setRowColStats(null);

    setTimeout(() => {
      try {
        if (inputMode === 'manual') {
          const parsedA = parseMatrixText(manualA);
          if ('error' in parsedA) { setErrorMsg(parsedA.error); setStatus('error'); return; }
          const parsedB = parseVectorText(manualB);
          if ('error' in parsedB) { setErrorMsg(parsedB.error); setStatus('error'); return; }
          const v = validateSystem(parsedA, parsedB);
          if (!v.valid) { setErrorMsg(`${v.error}: ${v.details || ''}`); setStatus('error'); return; }
          setA(parsedA);
          setB(parsedB);
          setMetrics(computeMatrixMetrics(parsedA));
        } else if (inputMode === 'python') {
          const lines = pythonInput.split('\n');
          const aLine = lines.find(l => l.trim().startsWith('A'));
          const bLine = lines.find(l => l.trim().startsWith('b'));
          if (!aLine || !bLine) { setErrorMsg('Missing A = … or b = … lines'); setStatus('error'); return; }
          const parsedA = parseMatrixText(aLine);
          if ('error' in parsedA) { setErrorMsg(parsedA.error); setStatus('error'); return; }
          const parsedB = parseVectorText(bLine);
          if ('error' in parsedB) { setErrorMsg(parsedB.error); setStatus('error'); return; }
          const v = validateSystem(parsedA, parsedB);
          if (!v.valid) { setErrorMsg(`${v.error}: ${v.details || ''}`); setStatus('error'); return; }
          setA(parsedA);
          setB(parsedB);
          setMetrics(computeMatrixMetrics(parsedA));
        } else {
          const { A: genA, b: genB } = generateSystem(configOverride || config);
          setA(genA);
          setB(genB);
          setMetrics(computeMatrixMetrics(genA));
        }
        setStatus('ready');
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Generation failed';
        setErrorMsg(msg);
        setStatus('error');
      }
    }, 40);
  }, [config, inputMode, manualA, manualB, pythonInput]);

  useEffect(() => {
    handleGenerate();
  }, []); // eslint-disable-line

  // ─── Solve ───
  const handleSolve = useCallback(() => {
    if (A.length === 0) return;
    setStatus('solving');
    setSolveAnimation(true);
    setErrorMsg(null);

    setTimeout(() => {
      try {
        let result: SolverResult;
        switch (solverMethod) {
          case 'lstsq':        result = solveLeastSquares(A, b); break;
          case 'jacobi':       result = solveJacobi(A, b); break;
          case 'gauss-seidel': result = solveGaussSeidel(A, b); break;
          default:             result = solveDirectLU(A, b);
        }
        setSolverResult(result);
        if (result.error) {
          setStatus(result.converged ? 'warning' : 'singular');
          setErrorMsg(result.error);
        } else {
          setStatus('solved');
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Solver execution error';
        setErrorMsg(msg);
        setStatus('error');
      }
      setTimeout(() => setSolveAnimation(false), 450);
    }, 80);
  }, [A, b, solverMethod]);

  // ─── Benchmark ───
  const handleBenchmark = useCallback(() => {
    if (A.length === 0) return;
    const results: { name: string; result: SolverResult }[] = [];
    results.push({ name: 'NumPy Direct (LU)',     result: solveDirectLU(A, b) });
    results.push({ name: 'Least Squares (lstsq)', result: solveLeastSquares(A, b) });
    if (A.length === A[0]?.length) {
      results.push({ name: 'Jacobi Iterative',       result: solveJacobi(A, b, 100, 1e-5) });
      results.push({ name: 'Gauss-Seidel',          result: solveGaussSeidel(A, b, 100, 1e-5) });
    }
    setBenchmarkResults(results);
    setActiveTab('compare');
  }, [A, b]);

  const Ax = useMemo(() => solverResult ? matrixVectorMultiply(A, solverResult.x) : null, [A, solverResult]);

  const conditionLevel = useMemo(() => {
    if (!metrics) return 'unknown';
    const k = metrics.conditionNumber;
    if (k < 100) return 'excellent';
    if (k < 1e4) return 'good';
    if (k < 1e8) return 'moderate';
    if (k < 1e12) return 'poor';
    return 'severe';
  }, [metrics]);

  const handleSelectRowCol = useCallback((type: 'row' | 'col', index: number) => {
    if (A.length === 0) return;
    setRowColStats(computeRowColStats(A, type, index));
  }, [A]);

  // Command palette items grouped
  const commandGroups = useMemo(() => [
    {
      category: 'Primary Actions',
      items: [
        { label: 'Solve System (Ax = b)', shortcut: '↵', icon: <Play size={14} className="text-[#68BA7F]" />, action: handleSolve },
        { label: 'Generate / Refresh System', shortcut: 'R', icon: <RefreshCw size={14} className="text-[#68BA7F]" />, action: handleGenerate },
        { label: 'Run Solvers Benchmark', shortcut: 'B', icon: <GitCompare size={14} className="text-[#DCD6F7]" />, action: handleBenchmark },
      ]
    },
    {
      category: 'Workspaces & Views',
      items: [
        { label: 'System Studio (Overview)', shortcut: '1', icon: <Grid3X3 size={14} />, action: () => setActiveTab('system') },
        { label: 'Matrix Heatmap & Sparsity', shortcut: '2', icon: <Eye size={14} />, action: () => { setMatrixView('heatmap'); setActiveTab('matrix'); } },
        { label: 'Vectors & Equation Verification', shortcut: '3', icon: <BarChart3 size={14} />, action: () => setActiveTab('solver') },
        { label: 'Residual & Condition Analysis', shortcut: '4', icon: <Activity size={14} />, action: () => setActiveTab('analysis') },
        { label: 'Python NumPy Code Script', shortcut: '5', icon: <Code2 size={14} />, action: () => setActiveTab('python') },
        { label: 'Compare Solvers Benchmark', shortcut: '6', icon: <GitCompare size={14} />, action: () => setActiveTab('compare') },
      ]
    },
    {
      category: 'Matrix Presets',
      items: PRESETS.slice(0, 6).map(p => ({
        label: `Load Preset: ${p.label}`,
        shortcut: '',
        icon: <Sparkles size={14} className="text-[#FFA896]" />,
        action: () => {
          setConfig(c => ({ ...c, preset: p.key }));
          setInputMode('generate');
          setTimeout(() => handleGenerate(), 20);
        }
      }))
    },
    {
      category: 'Settings & Modes',
      items: [
        { label: `Switch to ${themeMode === 'light' ? 'Dark' : 'Light'} Mode`, shortcut: 'T', icon: themeMode === 'light' ? <Moon size={14} /> : <Sun size={14} />, action: () => setThemeMode(m => m === 'light' ? 'dark' : 'light') },
        { label: 'Open Presentation Deck', shortcut: 'P', icon: <BookOpen size={14} />, action: () => { setPresentationMode(true); setPresentationStep(0); } },
        { label: 'Configure System Dimensions', shortcut: 'C', icon: <Sliders size={14} />, action: () => setShowConfigDrawer(true) },
      ]
    }
  ], [handleSolve, handleGenerate, handleBenchmark, themeMode]);

  const flatFilteredCommands = useMemo(() => {
    const q = commandFilter.toLowerCase().trim();
    if (!q) return commandGroups.flatMap(g => g.items);
    return commandGroups.flatMap(g => g.items).filter(item => item.label.toLowerCase().includes(q));
  }, [commandGroups, commandFilter]);

  // Keyboard navigation inside search
  const handleKeyDownSearch = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setCommandSelectedIndex(i => Math.min(flatFilteredCommands.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setCommandSelectedIndex(i => Math.max(0, i - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (flatFilteredCommands[commandSelectedIndex]) {
        flatFilteredCommands[commandSelectedIndex].action();
        setCommandOpen(false);
      }
    }
  };

  // ─── Radial Condition Gauge with requested pastel tones ───
  const ConditionGauge = () => {
    if (!metrics) return null;
    const value = metrics.conditionNumber;
    const logVal = Math.log10(Math.max(1, value));
    const ratio = Math.min(1, logVal / 16);
    const angle = -90 + ratio * 180;
    const cx2 = 60 + 38 * Math.cos((angle * Math.PI) / 180);
    const cy2 = 60 + 38 * Math.sin((angle * Math.PI) / 180);

    const gaugeColor = conditionLevel === 'excellent' || conditionLevel === 'good'
      ? '#68BA7F' // Pastel Sage
      : conditionLevel === 'moderate'
      ? '#E0A82E' // Pastel Butter
      : '#FFA896'; // Pastel Peach

    return (
      <div className="text-center flex flex-col items-center">
        <svg viewBox="0 0 120 72" className="w-full max-w-[160px]">
          <defs>
            <linearGradient id="chicGaugeGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#68BA7F" />
              <stop offset="50%" stopColor="#FFF0C8" />
              <stop offset="100%" stopColor="#FFA896" />
            </linearGradient>
          </defs>
          <path d="M 15 62 A 45 45 0 0 1 105 62" fill="none" stroke="var(--border-subtle)" strokeWidth="8" strokeLinecap="round" />
          <path
            d="M 15 62 A 45 45 0 0 1 105 62"
            fill="none"
            stroke="url(#chicGaugeGrad)"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${ratio * 141} 141`}
          />
          <line x1="60" y1="62" x2={cx2} y2={cy2} stroke={gaugeColor} strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="60" cy="62" r="5" fill={gaugeColor} />
          <text x="60" y="52" textAnchor="middle" fill="var(--text-main)" fontSize="11" fontFamily="JetBrains Mono, monospace" fontWeight="700">
            {value < 1e5 ? value.toFixed(1) : value.toExponential(1)}
          </text>
        </svg>
        <div className="flex items-center gap-1.5 mt-1">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: gaugeColor }} />
          <span className="text-xs font-bold uppercase tracking-wider" style={{ color: gaugeColor }}>
            {conditionLevel}
          </span>
        </div>
      </div>
    );
  };

  // ─── Presentation Mode ───
  if (presentationMode) {
    const slides = [
      <div key="s0" className="flex-1 flex flex-col items-center justify-center p-12 text-center gap-6">
        <span className="chic-badge badge-mint text-xs">LINEAR LAB · NUMPY LAPACK</span>
        <h1 className="text-5xl font-extrabold text-[var(--text-main)] tracking-tight max-w-3xl leading-tight">
          High-Performance Solution of <br />
          <span className="text-[#68BA7F]">Large Linear Systems</span>
        </h1>
        <div className="font-mono text-xl px-6 py-3.5 bg-[var(--bg-card)] border border-[var(--border-mid)] rounded-2xl shadow-sm text-[var(--text-body)]">
          <span className="text-[#68BA7F] font-bold">x</span> = np.linalg.<span className="text-[#68BA7F] font-bold">solve</span>(A, b)
        </div>
        <p className="text-base text-[var(--text-muted)] max-w-xl">
          Solving <strong className="text-[var(--text-main)]">Ax = b</strong> with numerical stability analysis, condition estimation, and multiple direct/iterative solvers.
        </p>
      </div>,

      <div key="s1" className="flex-1 flex flex-col items-center justify-center p-12 gap-8">
        <h2 className="text-3xl font-bold text-[var(--text-main)]">Matrix Heatmap & System Structure</h2>
        <div className="flex items-center gap-6 max-w-4xl w-full">
          <div className="flex-1 chic-card p-4">
            <div className="text-xs font-bold text-[#68BA7F] uppercase tracking-wider mb-2">Matrix A ({A.length}×{A[0]?.length || 0})</div>
            <MatrixHeatmapCanvas matrix={A} theme={themeMode} />
          </div>
          <div className="text-3xl font-mono text-[var(--text-faint)]">·</div>
          <div className="w-28 chic-card p-4 text-center">
            <div className="text-xs font-bold text-[var(--text-muted)] uppercase mb-2">Vector x</div>
            <div className="text-2xl font-mono font-bold text-[#68BA7F]">[{A[0]?.length || 0}×1]</div>
          </div>
          <div className="text-3xl font-mono text-[var(--text-faint)]">=</div>
          <div className="flex-1 chic-card p-4">
            <div className="text-xs font-bold text-[#FFA896] uppercase tracking-wider mb-2">RHS Vector b</div>
            {b.length > 0 && <VectorView vector={b} label="b" color="#FFA896" />}
          </div>
        </div>
      </div>,

      <div key="s2" className="flex-1 flex flex-col items-center justify-center p-12 gap-6">
        <h2 className="text-3xl font-bold text-[var(--text-main)]">Solution Verification</h2>
        {solverResult ? (
          <div className="flex flex-col items-center gap-6 w-full max-w-2xl">
            <div className="grid grid-cols-3 gap-4 w-full">
              <div className="stat-pill-card chic-card-mint">
                <span className="stat-label">Residual ‖Ax−b‖₂</span>
                <span className="stat-value text-[#68BA7F]">{solverResult.residualNorm.toExponential(3)}</span>
                <span className="stat-sub">High precision</span>
              </div>
              <div className="stat-pill-card chic-card-peach">
                <span className="stat-label">Solve Time</span>
                <span className="stat-value text-[#FFA896]">{solverResult.timeMs.toFixed(2)} ms</span>
                <span className="stat-sub">Optimized</span>
              </div>
              <div className="stat-pill-card chic-card-lavender">
                <span className="stat-label">Solution Norm ‖x‖₂</span>
                <span className="stat-value text-[#DCD6F7]">{vectorNorm(solverResult.x).toExponential(3)}</span>
                <span className="stat-sub">Computed</span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[#68BA7F] font-bold text-lg">
              <CheckCircle2 size={24} className="text-[#68BA7F]" />
              Numerical solution verified against residual threshold
            </div>
          </div>
        ) : (
          <p className="text-[var(--text-muted)]">Solve the system first to view solution results</p>
        )}
      </div>
    ];

    return (
      <div className="fixed inset-0 z-50 bg-[var(--bg-app)] flex flex-col">
        <div className="chic-bg"><div className="chic-bg-glow" /></div>
        <div className="relative z-10 flex items-center justify-between px-8 py-4 border-b border-[var(--border-subtle)] bg-[var(--bg-card)]/90 backdrop-blur-md">
          <span className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
            Presentation Mode · Slide {presentationStep + 1} of {slides.length}
          </span>
          <div className="flex items-center gap-2">
            <button
              className="btn btn-secondary text-xs"
              onClick={() => setPresentationStep(s => Math.max(0, s - 1))}
              disabled={presentationStep === 0}
            >
              <ChevronLeft size={14} /> Prev
            </button>
            <button
              className="btn btn-secondary text-xs"
              onClick={() => setPresentationStep(s => Math.min(slides.length - 1, s + 1))}
              disabled={presentationStep >= slides.length - 1}
            >
              Next <ChevronRight size={14} />
            </button>
            <button className="btn btn-ghost text-xs" onClick={() => setPresentationMode(false)}>
              <X size={14} /> Exit
            </button>
          </div>
        </div>
        <div className="relative z-10 flex-1 flex flex-col">{slides[presentationStep]}</div>
        <div className="relative z-10 flex justify-center gap-2 p-4">
          {slides.map((_, i) => (
            <button
              key={i}
              onClick={() => setPresentationStep(i)}
              className={`w-2.5 h-2.5 rounded-full transition-all ${
                i === presentationStep ? 'bg-[#68BA7F] scale-125' : 'bg-[var(--border-mid)]'
              }`}
            />
          ))}
        </div>
      </div>
    );
  }

  const n = A.length;
  const m = A[0]?.length || 0;
  const activePresetInfo = PRESETS.find(p => p.key === config.preset) || PRESETS[0];

  return (
    <div className="w-full h-full flex flex-col bg-[var(--bg-app)] overflow-hidden relative select-none">
      {/* Ambient Glow */}
      <div className="chic-bg"><div className="chic-bg-glow" /></div>

      {/* Top header: brand and primary actions only */}
      <header className="app-header relative z-30 px-6 bg-[var(--bg-card)]/95 backdrop-blur-md border-b border-[var(--border-subtle)] flex-shrink-0 shadow-sm">
        <div className="app-brand flex items-center gap-3">
          <button className="mobile-nav-trigger btn-icon" onClick={() => setMobileSidebarOpen(true)} title="Open navigation">
            <Menu size={19} />
          </button>
          <div className="flex items-center gap-3">
            <div className="app-logo w-10 h-10 rounded-xl bg-[#68BA7F] flex items-center justify-center text-white shadow-sm flex-shrink-0">
              <Cpu size={20} />
            </div>
            <div>
              <div className="flex items-center gap-1.5 leading-none">
                <span className="font-extrabold text-lg tracking-tight text-[var(--text-main)]">LINEAR</span>
                <span className="font-extrabold text-lg tracking-tight text-[#68BA7F]">LAB</span>
              </div>
              <span className="text-xs font-mono text-[var(--text-muted)] font-medium">
                Ax = b · NumPy
              </span>
            </div>
          </div>
        </div>

        <div className="app-controls flex items-center gap-2.5">
          <button
            onClick={handleSolve}
            disabled={A.length === 0}
            className="btn btn-primary header-solve shadow-sm"
          >
            <Play size={17} /> Solve
          </button>
          <button
            onClick={() => { setCommandOpen(true); setCommandFilter(''); setCommandSelectedIndex(0); }}
            className="search-trigger btn-icon"
            title="Spotlight Search (Ctrl+K)"
          >
            <Search size={17} />
            <span>Search</span>
            <kbd>Ctrl K</kbd>
          </button>
          <button
            onClick={() => setThemeMode(m => m === 'light' ? 'dark' : 'light')}
            className="btn-icon"
            title={`Switch to ${themeMode === 'light' ? 'Dark' : 'Light'} Mode`}
          >
            {themeMode === 'light' ? <Moon size={17} /> : <Sun size={17} />}
          </button>
          <button
            onClick={() => { setPresentationMode(true); setPresentationStep(0); }}
            className="btn-icon"
            title="Help and presentation slides"
          >
            <BookOpen size={17} />
          </button>
          <button
            onClick={() => setShowConfigDrawer(o => !o)}
            className={`btn-icon ${showConfigDrawer ? 'border-[#68BA7F] text-[#68BA7F] bg-[var(--c-sage-soft)]' : ''}`}
            title="Matrix System Settings"
          >
            <Sliders size={17} />
          </button>
        </div>
      </header>

      {/* ══════════════════════════════════════════════════════════════════
          MAIN STAGE & VIEWS (Clean, expansive viewport)
      ══════════════════════════════════════════════════════════════════ */}
      <div className="app-body relative z-10 flex-1 overflow-hidden flex">
        <aside className={`app-sidebar ${sidebarCollapsed ? 'collapsed' : ''} ${mobileSidebarOpen ? 'mobile-open' : ''}`} aria-label="Primary navigation">
          <div className="sidebar-heading">WORKSPACES</div>
          <nav className="sidebar-nav">
            {([
              ['system', 'Overview', <Grid3X3 size={21} />],
              ['matrix', 'Matrix', <Eye size={21} />],
              ['solver', 'Solution & Verification', <Play size={21} />],
              ['analysis', 'Residual Analysis', <Activity size={21} />],
              ['python', 'Python Lab', <Code2 size={21} />],
              ['compare', 'Benchmark', <GitCompare size={21} />],
            ] as [AppTab, string, React.ReactNode][]).map(([tab, label, icon]) => (
              <button key={tab} onClick={() => { setActiveTab(tab); setMobileSidebarOpen(false); }} className={`sidebar-nav-item ${activeTab === tab ? 'active' : ''}`} title={sidebarCollapsed ? label : undefined}>
                <span>{icon}</span><span className="sidebar-label">{label}</span>
              </button>
            ))}
          </nav>
          <div className={`sidebar-config ${systemConfigOpen ? 'open' : ''}`}>
            <button className="sidebar-section-toggle" onClick={() => setSystemConfigOpen(open => !open)} title="Toggle system configuration">
              <Sliders size={17} /><span className="sidebar-label">System Configuration</span><ChevronDown size={15} className="sidebar-config-chevron" />
            </button>
            {systemConfigOpen && !sidebarCollapsed && (
              <div className="sidebar-config-fields">
                <div className="sidebar-field relative" ref={presetDropdownRef}>
                  <span className="system-control-label">System</span>
                  <button onClick={() => setPresetDropdownOpen(o => !o)} className="sidebar-select preset-trigger" title="Select Matrix Preset">
                    <span className="text-[#68BA7F]">{activePresetInfo.icon}</span><span>{activePresetInfo.label}</span><ChevronDown size={14} className="ml-auto text-[var(--text-muted)]" />
                  </button>
                  {presetDropdownOpen && (
                    <div className="preset-menu sidebar-preset-menu absolute left-0 bottom-full mb-2 w-72 bg-[var(--bg-card)] border border-[var(--border-mid)] rounded-2xl shadow-2xl p-2 z-50 animate-fade-in">
                      <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Select Matrix Preset</div>
                      <div className="max-h-80 overflow-y-auto flex flex-col gap-1">
                        {PRESETS.map(p => <button key={p.key} onClick={() => { const nextConfig = { ...config, preset: p.key }; setConfig(nextConfig); setInputMode('generate'); setPresetDropdownOpen(false); setTimeout(() => handleGenerate(nextConfig), 20); }} className={`flex items-center justify-between p-2 rounded-xl text-left text-xs transition-colors ${config.preset === p.key && inputMode === 'generate' ? 'bg-[var(--c-sage-soft)] text-[#68BA7F] font-bold' : 'text-[var(--text-body)] hover:bg-[var(--bg-hover)]'}`}><div className="flex items-center gap-2.5"><span className="p-1 rounded-lg bg-[var(--bg-card-subtle)] border border-[var(--border-subtle)]">{p.icon}</span><div><div>{p.label}</div><div className="text-[10px] font-normal text-[var(--text-muted)]">{p.desc}</div></div></div>{config.preset === p.key && inputMode === 'generate' && <Check size={14} className="text-[#68BA7F]" />}</button>)}
                      </div>
                    </div>
                  )}
                </div>
                <div className="sidebar-field"><span className="system-control-label">Size</span><div className="sidebar-size-control"><button onClick={() => { const nextSize = Math.max(2, config.size - 1); const nextConfig = { ...config, size: nextSize, rows: nextSize, cols: nextSize }; setConfig(nextConfig); setTimeout(() => handleGenerate(nextConfig), 20); }} className="dimension-step" title="Decrease Matrix Dimension">−</button><span>{n}×{m}</span><button onClick={() => { const nextSize = Math.min(100, config.size + 1); const nextConfig = { ...config, size: nextSize, rows: nextSize, cols: nextSize }; setConfig(nextConfig); setTimeout(() => handleGenerate(nextConfig), 20); }} className="dimension-step" title="Increase Matrix Dimension">+</button></div></div>
                <div className="sidebar-field"><label className="system-control-label" htmlFor="sidebar-solver-select">Solver</label><select id="sidebar-solver-select" value={solverMethod} onChange={e => setSolverMethod(e.target.value as SolverType)} className="sidebar-select text-xs font-semibold text-[var(--text-body)] bg-[var(--bg-card-subtle)] border border-[var(--border-subtle)] rounded-xl cursor-pointer outline-none">{SOLVER_OPTIONS.map(s => <option key={s.key} value={s.key}>{s.label}</option>)}</select></div>
                <div className="sidebar-field"><span className="system-control-label">Status</span><div className="sidebar-status chic-badge badge-mint"><span className="w-2 h-2 rounded-full bg-[#68BA7F] animate-pulse" /><span className="capitalize">{status === 'solved' ? 'Solved ✓' : status}</span></div></div>
              </div>
            )}
          </div>
          <button className="sidebar-collapse" onClick={() => setSidebarCollapsed(c => !c)} title={sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'}>
            {sidebarCollapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}<span className="sidebar-label">{sidebarCollapsed ? 'Expand' : 'Collapse'}</span>
          </button>
        </aside>
        <main className="app-main relative z-10 flex-1 overflow-hidden flex flex-col">
        {/* Solving Micro-Animation Overlay */}
        {solveAnimation && (
          <div className="absolute inset-0 z-40 bg-[var(--bg-app)]/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
            <div className="w-12 h-12 rounded-full border-4 border-[var(--c-mint)] border-t-[#68BA7F] animate-spin-chic" />
            <div className="text-base font-bold text-[var(--text-main)]">Solving System via LAPACK…</div>
            <div className="font-mono text-xs text-[var(--text-muted)]">x = np.linalg.solve(A, b)</div>
          </div>
        )}

        {/* Global Error Notice if any */}
        {errorMsg && (
          <div className="mx-6 mt-3 px-4 py-2.5 rounded-xl bg-[var(--c-peach-soft)] border border-[var(--c-peach-border)] text-[#FFA896] text-xs flex items-center justify-between">
            <div className="flex items-center gap-2 font-medium">
              <AlertTriangle size={15} />
              <span>{errorMsg}</span>
            </div>
            <button onClick={() => setErrorMsg(null)} className="hover:opacity-70">
              <X size={14} />
            </button>
          </div>
        )}

        {/* ─── TAB 1: SYSTEM STUDIO (Overview) ─── */}
        {activeTab === 'system' && (
          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">
            {/* Top Metric Cards Ribbon */}
            {n > 0 && (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <div className="stat-pill-card chic-card-mint">
                  <span className="stat-label">System Size</span>
                  <span className="stat-value text-[#68BA7F]">{n}×{m}</span>
                  <span className="stat-sub">{n * m} matrix elements</span>
                </div>

                <div className="stat-pill-card chic-card-peach">
                  <span className="stat-label">Condition κ(A)</span>
                  <span className="stat-value text-[#FFA896]">
                    {metrics ? (metrics.conditionNumber > 1e6 ? metrics.conditionNumber.toExponential(1) : metrics.conditionNumber.toFixed(1)) : '—'}
                  </span>
                  <span className="stat-sub capitalize">{conditionLevel}</span>
                </div>

                <div className="stat-pill-card chic-card-lavender">
                  <span className="stat-label">Sparsity</span>
                  <span className="stat-value text-[#DCD6F7]">
                    {metrics ? `${(metrics.sparsity * 100).toFixed(1)}%` : '—'}
                  </span>
                  <span className="stat-sub">{metrics?.nonZeroCount || 0} non-zeros</span>
                </div>

                <div className="stat-pill-card chic-card-butter">
                  <span className="stat-label">Frobenius ‖A‖F</span>
                  <span className="stat-value text-[#FFF0C8]">
                    {metrics ? fmt(metrics.frobeniusNorm, 2) : '—'}
                  </span>
                  <span className="stat-sub">Matrix norm</span>
                </div>

                <div className="stat-pill-card">
                  <span className="stat-label">Symmetry</span>
                  <span className="stat-value" style={{ color: metrics?.isSymmetric ? '#68BA7F' : 'var(--text-body)' }}>
                    {metrics?.isSymmetric ? 'Yes (A=Aᵀ)' : 'No'}
                  </span>
                  <span className="stat-sub">{metrics?.isPositiveDefinite ? 'Pos. Definite' : 'General'}</span>
                </div>

                <div className="stat-pill-card chic-card-mint">
                  <span className="stat-label">Residual ‖Ax−b‖₂</span>
                  <span className="stat-value text-[#68BA7F]">
                    {solverResult ? solverResult.residualNorm.toExponential(2) : '—'}
                  </span>
                  <span className="stat-sub">{solverResult ? `${solverResult.timeMs.toFixed(2)} ms` : 'Click Solve'}</span>
                </div>
              </div>
            )}

            <div className="overview-flow">
              <section className="overview-matrix-card chic-card">
                <div className="overview-card-header">
                  <div>
                    <span className="section-kicker">MATRIX A</span>
                    <h3>Heatmap & Cell Inspector</h3>
                    <p>Inspect entries, rows, and columns directly on the matrix.</p>
                  </div>
                  <button onClick={() => setActiveTab('matrix')} className="btn btn-secondary overview-deep-view">
                    Deep View <ArrowUpRight size={16} />
                  </button>
                </div>
                <div className="matrix-canvas-frame">
                  <MatrixHeatmapCanvas matrix={A} onSelectRowCol={handleSelectRowCol} theme={themeMode} />
                </div>
              </section>

              <section className="overview-vector-card chic-card">
                <div className="overview-card-header compact">
                  <div>
                    <span className="section-kicker section-kicker-peach">VECTOR b</span>
                    <h3>Right-Hand Side</h3>
                    <p>The input vector used to define <strong>Ax = b</strong>.</p>
                  </div>
                </div>
                <VectorView vector={b} label="b" color="#FFA896" />
              </section>

              <div className="overview-support-grid">
                <section className="condition-panel chic-card p-5 flex items-center justify-between gap-6">
                  <div className="flex-1">
                    <span className="section-kicker">CONDITION NUMBER</span>
                    <h3 className="text-xl font-bold text-[var(--text-main)] mt-1">
                      κ(A) = {metrics ? (metrics.conditionNumber > 1e6 ? metrics.conditionNumber.toExponential(2) : metrics.conditionNumber.toFixed(2)) : '1.0'}
                    </h3>
                    <p className="text-sm text-[var(--text-muted)] mt-2">
                      {conditionLevel === 'excellent' ? 'Ideal conditioning: inversion is perfectly stable.' :
                       conditionLevel === 'good' ? 'Well-conditioned system: standard round-off bounds apply.' :
                       conditionLevel === 'moderate' ? 'Moderate condition: some sensitivity to perturbations.' :
                       'Ill-conditioned: matrix is near-singular! Expect precision loss.'}
                    </p>
                  </div>
                  <div className="w-36 flex-shrink-0"><ConditionGauge /></div>
                </section>

                <section className="solution-status-card chic-card chic-card-mint p-5">
                  {solverResult ? (
                    <>
                      <div className="solution-status-heading">
                        <div><span className="section-kicker section-kicker-sage">SOLUTION</span><h3>Solution found</h3></div>
                        <CheckCircle2 size={22} className="text-[#68BA7F]" />
                      </div>
                      <div className="solution-status-metrics"><span>{solverResult.timeMs.toFixed(2)} ms</span><span>‖r‖₂ = {solverResult.residualNorm.toExponential(2)}</span></div>
                      <div className="solution-preview">x = [{solverResult.x.slice(0, 5).map(v => v.toFixed(3)).join(', ')}{solverResult.x.length > 5 ? ', …' : ''}]</div>
                    </>
                  ) : (
                    <div className="solution-empty"><span className="section-kicker section-kicker-sage">SOLUTION</span><h3>Ready to solve</h3><p>Run the selected solver to calculate x and verify the residual.</p></div>
                  )}
                </section>
              </div>
            </div>

            {/* Row / Col Selected Stats if clicked */}
            {rowColStats && (
              <div className="chic-card p-4 animate-fade-in">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--text-main)]">
                    Selected {rowColStats.type === 'row' ? 'Row' : 'Column'} #{rowColStats.index + 1} Statistics
                  </h4>
                  <button onClick={() => setRowColStats(null)} className="text-[var(--text-muted)] hover:text-[var(--text-main)]">
                    <X size={14} />
                  </button>
                </div>
                <div className="grid grid-cols-5 gap-3">
                  {[
                    ['Min', fmt(rowColStats.min)],
                    ['Max', fmt(rowColStats.max)],
                    ['Mean', fmt(rowColStats.mean)],
                    ['Euclidean ‖v‖₂', fmt(rowColStats.norm)],
                    ['Non-Zeros', `${rowColStats.nonZeroCount} / ${rowColStats.type === 'row' ? m : n}`],
                  ].map(([k, v]) => (
                    <div key={k} className="p-2.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-subtle)] text-center">
                      <div className="text-[10.5px] uppercase font-semibold text-[var(--text-muted)]">{k}</div>
                      <div className="font-mono text-sm font-bold text-[var(--text-main)] mt-1">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 2: MATRIX EXPLORER ─── */}
        {activeTab === 'matrix' && (
          <div className="matrix-workspace flex-1 overflow-hidden flex flex-col p-6 gap-4">
            {/* View Selector & Metrics */}
            <div className="flex items-center justify-between px-5 py-3 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-sm">
              <div className="flex items-center gap-3">
                <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)]">Display Mode:</span>
                <div className="chic-tab-container">
                  {([
                    ['heatmap', 'Heatmap Canvas', <Eye size={13} />],
                    ['sparsity', 'Sparsity Dot Pattern', <Sparkles size={13} />],
                    ['values', 'Raw Matrix Values', <Grid3X3 size={13} />]
                  ] as [MatrixViewMode, string, React.ReactNode][]).map(([v, label, icon]) => (
                    <button
                      key={v}
                      onClick={() => setMatrixView(v)}
                      className={`chic-tab-btn ${matrixView === v ? 'active' : ''}`}
                    >
                      {icon} {label}
                    </button>
                  ))}
                </div>
              </div>

              {metrics && (
                <div className="flex items-center gap-4 text-xs font-mono">
                  <span className="text-[var(--text-muted)]">Sparsity: <strong className="text-[#68BA7F]">{(metrics.sparsity * 100).toFixed(1)}%</strong></span>
                  <span className="text-[var(--text-muted)]">Non-zeros: <strong className="text-[var(--text-main)]">{metrics.nonZeroCount}</strong></span>
                  <span className="text-[var(--text-muted)]">Rank: <strong className="text-[var(--text-main)]">{metrics.rank}</strong></span>
                </div>
              )}
            </div>

            {/* View Render Area */}
            <div className="matrix-render-area flex-1 min-h-[420px] chic-card p-4 overflow-hidden">
              {matrixView === 'heatmap' && (
                <MatrixHeatmapCanvas matrix={A} onSelectRowCol={handleSelectRowCol} theme={themeMode} />
              )}
              {matrixView === 'sparsity' && (
                <SparsityCanvas matrix={A} theme={themeMode} />
              )}
              {matrixView === 'values' && (
                <div className="h-full overflow-auto p-4 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-subtle)] font-mono text-xs">
                  {A.map((row, i) => (
                    <div key={i} className="flex items-center gap-4 py-1 border-b border-[var(--border-subtle)] hover:bg-[var(--bg-card)] px-2">
                      <span className="w-8 text-[var(--text-muted)] font-bold select-none">[{i}]</span>
                      <div className="flex items-center gap-6 flex-wrap">
                        {row.map((val, j) => (
                          <span
                            key={j}
                            className="w-24 text-right"
                            style={{
                              color: val > 0 ? '#68BA7F' : val < 0 ? '#FFA896' : 'var(--text-muted)',
                              fontWeight: Math.abs(val) > 0 ? 600 : 400
                            }}
                          >
                            {fmt(val, 4)}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── TAB 3: SOLVER & VECTORS ─── */}
        {activeTab === 'solver' && (
          <div className="solver-workspace flex-1 overflow-y-auto p-6 flex flex-col gap-5">
            {/* Equation Math Header */}
            <div className="chic-card chic-card-mint p-5 text-center">
              <span className="text-xs font-bold uppercase tracking-widest text-[#68BA7F]">Linear Equation System</span>
              <div className="flex items-center justify-center gap-5 my-2 font-mono text-3xl font-extrabold text-[var(--text-main)]">
                <span className="text-[#68BA7F]">A</span>
                <span className="text-[#68BA7F]">x</span>
                <span className="text-[var(--text-faint)]">=</span>
                <span className="text-[#FFA896]">b</span>
              </div>
              <p className="text-xs font-mono text-[var(--text-muted)]">
                Matrix A [{n}×{m}] · Vector x [{m}×1] = Vector b [{n}×1]
              </p>
            </div>

            {/* Solution Vector Components + Verification Table */}
            {solverResult ? (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                <div className="lg:col-span-6 chic-card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-[#68BA7F]" />
                      <h4 className="text-sm font-bold text-[var(--text-main)]">Solution Vector x</h4>
                    </div>
                    <span className="text-xs font-mono text-[var(--text-muted)]">
                      {solverResult.x.length} components
                    </span>
                  </div>
                  <VectorView vector={solverResult.x} label="x" color="#68BA7F" />
                </div>

                <div className="lg:col-span-6 chic-card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-[#FFA896]" />
                      <h4 className="text-sm font-bold text-[var(--text-main)]">Equation Verification (Ax vs b)</h4>
                    </div>
                    <span className="text-xs font-mono text-[#68BA7F] font-bold">
                      ‖Ax - b‖₂ = {solverResult.residualNorm.toExponential(3)}
                    </span>
                  </div>
                  {Ax && <VerificationView Ax={Ax} b={b} />}
                </div>
              </div>
            ) : (
              <div className="chic-card p-12 text-center flex flex-col items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-[var(--c-sage-soft)] text-[#68BA7F] flex items-center justify-center">
                  <Play size={22} />
                </div>
                <h4 className="text-base font-bold text-[var(--text-main)]">System is ready to solve</h4>
                <p className="text-xs text-[var(--text-muted)] max-w-sm">
                  Click the Solve button below to execute the numerical LAPACK solver.
                </p>
                <button onClick={handleSolve} className="btn btn-primary text-sm mt-2">
                  <Play size={15} /> Solve with NumPy
                </button>
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 4: RESIDUAL ANALYSIS ─── */}
        {activeTab === 'analysis' && (
          <div className="analysis-workspace flex-1 overflow-y-auto p-6 flex flex-col gap-5">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-5 chic-card p-5 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-[var(--text-main)]">Condition Number κ(A)</h4>
                  <span className={`chic-badge ${conditionLevel === 'good' || conditionLevel === 'excellent' ? 'badge-mint' : 'badge-peach'}`}>
                    {conditionLevel}
                  </span>
                </div>
                <ConditionGauge />
                <div className="p-3.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-subtle)] text-xs leading-relaxed text-[var(--text-muted)]">
                  The condition number <strong className="text-[var(--text-main)]">κ(A) = ‖A‖ · ‖A⁻¹‖</strong> measures the sensitivity of the solution <strong className="text-[var(--text-main)]">x</strong> to small perturbations in the matrix <strong className="text-[var(--text-main)]">A</strong> or vector <strong className="text-[var(--text-main)]">b</strong>. A condition number near 1.0 represents optimal numerical stability.
                </div>
              </div>

              <div className="lg:col-span-7 chic-card p-5">
                <h4 className="text-sm font-bold text-[var(--text-main)] mb-4">Matrix Properties & Norms</h4>
                {metrics ? (
                  <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                    {[
                      ['Frobenius Norm ‖A‖F', fmt(metrics.frobeniusNorm)],
                      ['L1 Operator Norm ‖A‖₁', fmt(metrics.l2Norm)],
                      ['Min Element', fmt(metrics.min)],
                      ['Max Element', fmt(metrics.max)],
                      ['Non-Zero Elements', `${metrics.nonZeroCount} / ${metrics.totalCount}`],
                      ['Sparsity Ratio', `${(metrics.sparsity * 100).toFixed(2)}%`],
                      ['Rank Estimate', String(metrics.rank)],
                      ['Determinant', metrics.determinant !== null ? fmt(metrics.determinant) : 'N/A (Rectangular)'],
                    ].map(([label, value]) => (
                      <div key={label} className="p-3 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-subtle)] flex justify-between items-center">
                        <span className="text-[var(--text-muted)] font-sans text-xs">{label}</span>
                        <span className="font-bold text-[var(--text-main)]">{value}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-[var(--text-muted)]">No matrix generated.</p>
                )}
              </div>
            </div>

            {solverResult?.history && solverResult.history.length > 1 && (
              <div className="chic-card p-5">
                <h4 className="text-sm font-bold text-[var(--text-main)] mb-3">
                  Iterative Residual Convergence log₁₀(‖r‖₂)
                </h4>
                <ConvergenceChart history={solverResult.history} />
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 5: PYTHON LAB ─── */}
        {activeTab === 'python' && (
          <div className="python-workspace flex-1 overflow-y-auto p-6">
            <PythonLab A={A} b={b} solverMethod={solverMethod} />
          </div>
        )}

        {/* ─── TAB 6: COMPARE SOLVERS BENCHMARK ─── */}
        {activeTab === 'compare' && (
          <div className="benchmark-workspace flex-1 overflow-y-auto p-6 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-[var(--text-main)]">Solver Benchmark Comparison</h3>
                <p className="text-xs text-[var(--text-muted)]">Direct LU vs. Least-Squares vs. Iterative solvers</p>
              </div>
              <button onClick={handleBenchmark} className="btn btn-primary text-xs py-1.5 px-3.5">
                <GitCompare size={14} /> Run All Solvers
              </button>
            </div>

            {benchmarkResults.length === 0 ? (
              <div className="chic-card p-12 text-center flex flex-col items-center gap-3">
                <GitCompare size={32} className="text-[#68BA7F]" />
                <h4 className="text-sm font-bold text-[var(--text-main)]">No benchmark results yet</h4>
                <p className="text-xs text-[var(--text-muted)]">
                  Click the button above to compare solve time and residual precision across all numerical methods.
                </p>
                <button onClick={handleBenchmark} className="btn btn-secondary text-xs mt-2">
                  Run Benchmark Now
                </button>
              </div>
            ) : (
              <div className="benchmark-grid grid grid-cols-1 gap-3">
                {benchmarkResults.map(({ name, result }, idx) => {
                  const maxTime = Math.max(...benchmarkResults.map(r => r.result.timeMs), 0.01);
                  const timePct = Math.max(8, (result.timeMs / maxTime) * 100);
                  const isFastest = result.timeMs === Math.min(...benchmarkResults.map(r => r.result.timeMs));

                  return (
                    <div key={idx} className="benchmark-card chic-card p-4 flex flex-col gap-3">
                      <div className="benchmark-card-header">
                        <div className="benchmark-card-title">
                          <h4 className="text-base font-bold text-[var(--text-main)]">{name}</h4>
                          {isFastest && (
                            <span className="chic-badge badge-mint benchmark-badge">Fastest ⚡</span>
                          )}
                        </div>
                        <div className="benchmark-metrics">
                          <span>
                            <span className="benchmark-metric-label">Residual</span><strong className="text-[#FFA896]">{result.residualNorm.toExponential(2)}</strong>
                          </span>
                          <span>
                            <span className="benchmark-metric-label">Time</span><strong className="text-[#68BA7F]">{result.timeMs.toFixed(3)} ms</strong>
                          </span>
                        </div>
                      </div>

                      <div className="benchmark-progress w-full h-3 rounded-full bg-[var(--border-subtle)] overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${timePct}%`,
                            background: isFastest ? '#68BA7F' : '#FFA896'
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
        </main>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          SIDEBAR CONFIG DRAWER (Settings for dimensions, manual entry)
      ══════════════════════════════════════════════════════════════════ */}
      {showConfigDrawer && (
        <div
          className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex justify-end"
          onClick={e => { if (e.target === e.currentTarget) setShowConfigDrawer(false); }}
        >
          <div className="w-full max-w-md h-full bg-[var(--bg-card)] border-l border-[var(--border-subtle)] shadow-2xl flex flex-col animate-fade-in">
            <div className="p-5 border-b border-[var(--border-subtle)] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders size={18} className="text-[#68BA7F]" />
                <h3 className="text-base font-bold text-[var(--text-main)]">System Configuration</h3>
              </div>
              <button onClick={() => setShowConfigDrawer(false)} className="btn-icon">
                <X size={15} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-5">
              <div>
                <label className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] block mb-2">
                  Input Mode
                </label>
                <div className="chic-tab-container w-full">
                  {(['generate', 'manual', 'python'] as const).map(m => (
                    <button
                      key={m}
                      onClick={() => setInputMode(m)}
                      className={`chic-tab-btn flex-1 justify-center capitalize ${inputMode === m ? 'active' : ''}`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>

              {inputMode === 'generate' && (
                <div className="flex flex-col gap-4">
                  <div>
                    <div className="flex justify-between items-center mb-1.5 text-xs font-semibold text-[var(--text-body)]">
                      <span>Matrix Dimension (N×N)</span>
                      <span className="font-mono font-bold text-[#68BA7F]">{config.size}×{config.size}</span>
                    </div>
                    <input
                      type="range"
                      min={2}
                      max={100}
                      value={config.size}
                      onChange={e => setConfig(c => ({ ...c, size: +e.target.value, rows: +e.target.value, cols: +e.target.value }))}
                    />
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-[var(--text-body)] block mb-1">Random Seed</label>
                    <input
                      type="number"
                      value={config.seed}
                      onChange={e => setConfig(c => ({ ...c, seed: +e.target.value }))}
                      className="w-full"
                    />
                  </div>
                </div>
              )}

              {inputMode === 'manual' && (
                <div className="flex flex-col gap-3">
                  <div>
                    <label className="text-xs font-semibold text-[var(--text-body)] block mb-1">Matrix A Values</label>
                    <textarea
                      rows={5}
                      value={manualA}
                      onChange={e => setManualA(e.target.value)}
                      className="w-full font-mono text-xs"
                      placeholder="e.g.&#10;3 2 -1&#10;2 -2 4&#10;-1 0.5 -1"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-[var(--text-body)] block mb-1">Vector b Values</label>
                    <textarea
                      rows={3}
                      value={manualB}
                      onChange={e => setManualB(e.target.value)}
                      className="w-full font-mono text-xs"
                      placeholder="e.g.&#10;1&#10;-2&#10;0"
                    />
                  </div>
                </div>
              )}

              {inputMode === 'python' && (
                <div>
                  <label className="text-xs font-semibold text-[var(--text-body)] block mb-1">NumPy Array Code</label>
                  <textarea
                    rows={6}
                    value={pythonInput}
                    onChange={e => setPythonInput(e.target.value)}
                    className="w-full font-mono text-xs"
                  />
                </div>
              )}
            </div>

            <div className="p-5 border-t border-[var(--border-subtle)] flex items-center gap-3">
              <button
                onClick={() => {
                  handleGenerate();
                  setShowConfigDrawer(false);
                }}
                className="btn btn-primary flex-1"
              >
                Apply & Generate
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════
          SPOTLIGHT RAYCAST-STYLE SEARCH MODAL (Ctrl+K)
      ══════════════════════════════════════════════════════════════════ */}
      {commandOpen && (
        <div
          className="command-overlay fixed inset-0 z-50 bg-black/60 backdrop-blur-md flex items-start justify-center pt-20 px-4"
          onClick={e => { if (e.target === e.currentTarget) setCommandOpen(false); }}
        >
          <div className="command-palette w-full max-w-xl bg-[var(--bg-card)] border border-[var(--border-mid)] rounded-2xl shadow-2xl overflow-hidden animate-fade-in flex flex-col">
            {/* Search Input */}
            <div className="command-palette-search flex items-center gap-3 px-5 py-4 border-b border-[var(--border-subtle)] bg-[var(--bg-card-subtle)]">
              <Search size={21} className="text-[#68BA7F] flex-shrink-0" />
              <input
                ref={cmdInputRef}
                value={commandFilter}
                onChange={e => {
                  setCommandFilter(e.target.value);
                  setCommandSelectedIndex(0);
                }}
                onKeyDown={handleKeyDownSearch}
                placeholder="Search actions, views, matrix presets, or solvers…"
                className="command-search-input flex-1 border-none outline-none text-sm text-[var(--text-main)] bg-transparent font-medium"
              />
              <span className="command-close-hint px-2 py-0.5 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[10px] font-mono text-[var(--text-muted)]">
                ESC to close
              </span>
            </div>

            {/* Results List */}
            <div className="command-results max-h-80 overflow-y-auto p-2 flex flex-col gap-1">
              {flatFilteredCommands.map((cmd, idx) => {
                const isSelected = idx === commandSelectedIndex;
                return (
                  <button
                    key={idx}
                    onClick={() => {
                      cmd.action();
                      setCommandOpen(false);
                    }}
                    onMouseEnter={() => setCommandSelectedIndex(idx)}
                    className={`command-result-row w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-left text-xs font-semibold transition-all ${
                      isSelected
                        ? 'bg-[var(--c-sage-soft)] text-[#68BA7F] shadow-sm'
                        : 'text-[var(--text-body)] hover:bg-[var(--bg-hover)]'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="command-result-icon p-1 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)]">
                        {cmd.icon}
                      </span>
                      <span>{cmd.label}</span>
                    </div>
                    {cmd.shortcut && (
                      <kbd className="command-shortcut px-1.5 py-0.5 rounded bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[10px] font-mono text-[var(--text-muted)]">
                        {cmd.shortcut}
                      </kbd>
                    )}
                  </button>
                );
              })}

              {flatFilteredCommands.length === 0 && (
                <div className="p-8 text-center text-xs text-[var(--text-muted)]">
                  No matching commands found for "{commandFilter}"
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="command-palette-footer px-5 py-2.5 border-t border-[var(--border-subtle)] bg-[var(--bg-card-subtle)] text-[11px] text-[var(--text-muted)] flex items-center justify-between font-mono">
              <div className="flex items-center gap-3">
                <span>↑↓ navigate</span>
                <span>↵ select</span>
              </div>
              <span className="text-[#68BA7F] font-sans font-medium">Linear Lab Spotlight</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
