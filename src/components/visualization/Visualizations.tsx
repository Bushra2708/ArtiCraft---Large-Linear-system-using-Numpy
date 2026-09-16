import React, { useState } from 'react';
import type { Vector, IterationStep } from '../../math/types';
import { BarChart3, Grid3X3, Thermometer, Search } from 'lucide-react';

/* ==================== VECTOR VIEW ==================== */
interface VectorViewProps {
  vector: Vector;
  label: string;
  color?: string;
  glowClass?: string;
}

export const VectorView: React.FC<VectorViewProps> = ({ vector, label, color = '#68BA7F' }) => {
  const [mode, setMode] = useState<'bar' | 'heatmap' | 'table'>('bar');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const pageSize = 50;

  const absMax = Math.max(...vector.map(Math.abs), 1e-6);
  const norm = Math.sqrt(vector.reduce((s, v) => s + v * v, 0));
  const minVal = Math.min(...vector);
  const maxVal = Math.max(...vector);

  const filteredIndices = search
    ? vector.map((_, i) => i).filter(i => i.toString().includes(search) || vector[i].toFixed(6).includes(search))
    : vector.map((_, i) => i);

  const pageIndices = filteredIndices.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(filteredIndices.length / pageSize);

  return (
    <div className="w-full flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
          <h4 className="vector-heading text-lg font-bold">Vector {label}</h4>
        </div>
        <div className="vector-mode-switch flex items-center gap-1 rounded-xl p-1 border">
          {([['bar', BarChart3, 'Bar Chart'], ['heatmap', Thermometer, 'Intensity'], ['table', Grid3X3, 'Table']] as const).map(([m, Icon, title]) => (
            <button
              key={m}
              onClick={() => setMode(m as 'bar' | 'heatmap' | 'table')}
              title={title}
              className={`p-1.5 rounded-lg text-xs transition-all ${
                mode === m
                  ? 'vector-mode-active shadow-sm font-semibold'
                  : 'vector-mode-idle'
              }`}
            >
              <Icon size={14} />
            </button>
          ))}
        </div>
      </div>

      {/* Vector Stats Strip */}
      <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
        <span className="vector-stat vector-stat-neutral px-2.5 py-1 rounded-lg border">
          dim: <strong>{vector.length}</strong>
        </span>
        <span className="vector-stat vector-stat-sage px-2.5 py-1 rounded-lg border">
          ‖v‖₂: <strong>{norm.toExponential(3)}</strong>
        </span>
        <span className="vector-stat vector-stat-peach px-2.5 py-1 rounded-lg border">
          min: <strong>{minVal.toFixed(3)}</strong>
        </span>
        <span className="vector-stat vector-stat-sage px-2.5 py-1 rounded-lg border">
          max: <strong>{maxVal.toFixed(3)}</strong>
        </span>
      </div>

      {vector.length > 15 && (
        <div className="relative">
          <Search size={13} className="absolute left-3 top-2.5 viz-muted" />
          <input
            type="text"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search index or value…"
            className="vector-search w-full pl-8 pr-3 py-1.5 border rounded-xl text-xs focus:outline-none focus:border-[#68BA7F]"
          />
        </div>
      )}

      {/* Vector Display Container */}
      <div className="vector-display max-h-[280px] overflow-y-auto rounded-xl border">
        {mode === 'bar' && pageIndices.map(i => {
          const v = vector[i];
          const ratio = Math.abs(v) / absMax;
          const isPos = v >= 0;
          return (
            <div
              key={i}
              className="vector-row flex items-center gap-3 px-3 py-1.5 border-b text-xs font-mono transition-colors"
            >
              <span className="w-8 viz-muted text-right font-medium">
                x<sub>{i + 1}</sub>
              </span>
              <div className="vector-bar-track flex-1 h-3.5 rounded-full overflow-hidden relative">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${ratio * 100}%`,
                    background: isPos ? '#68BA7F' : '#FFA896'
                  }}
                />
              </div>
              <span
                className="vector-value w-24 text-right font-semibold"
                style={{ color: isPos ? '#276239' : '#DE6D55' }}
              >
                {v.toFixed(5)}
              </span>
            </div>
          );
        })}

        {mode === 'heatmap' && (
          <div className="flex flex-wrap gap-1 p-3">
            {pageIndices.map(i => {
              const v = vector[i];
              const ratio = v / absMax;
              const isPos = ratio >= 0;
              const bg = isPos
                ? `rgba(104, 186, 127, ${Math.max(0.2, Math.abs(ratio))})`
                : `rgba(255, 168, 150, ${Math.max(0.2, Math.abs(ratio))})`;

              return (
                <div
                  key={i}
                  className="vector-heatmap-cell w-6 h-6 rounded-md cursor-pointer relative group flex items-center justify-center text-[9px] font-mono font-bold border"
                  style={{ backgroundColor: bg }}
                  title={`x${i + 1} = ${v.toFixed(5)}`}
                >
                  <div className="vector-tooltip absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block z-20 px-2 py-1 rounded-lg text-[10px] font-mono whitespace-nowrap shadow-lg">
                    x<sub>{i + 1}</sub> = {v.toFixed(6)}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {mode === 'table' && (
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="vector-table-head border-b">
                <th className="px-3 py-2 text-left">Index</th>
                <th className="px-3 py-2 text-right">Value</th>
                <th className="px-3 py-2 text-right">|Value|</th>
              </tr>
            </thead>
            <tbody>
              {pageIndices.map(i => (
                <tr key={i} className="vector-table-row border-b transition-colors">
                  <td className="px-3 py-1.5 viz-muted">x<sub>{i + 1}</sub></td>
                  <td
                    className="px-3 py-1.5 text-right font-semibold"
                    style={{ color: vector[i] >= 0 ? '#276239' : '#DE6D55' }}
                  >
                    {vector[i].toFixed(6)}
                  </td>
                  <td className="px-3 py-1.5 text-right viz-muted">
                    {Math.abs(vector[i]).toExponential(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs viz-muted px-1">
          <button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
            className="vector-pager px-2.5 py-1 border rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium"
          >
            ← Prev
          </button>
          <span>Page {page + 1} of {totalPages}</span>
          <button
            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
            disabled={page >= totalPages - 1}
            className="vector-pager px-2.5 py-1 border rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
};

/* ==================== CONVERGENCE CHART ==================== */
interface ConvergenceChartProps {
  history: IterationStep[];
  width?: number;
  height?: number;
}

export const ConvergenceChart: React.FC<ConvergenceChartProps> = ({ history, width = 500, height = 220 }) => {
  if (history.length < 2) {
    return <div className="viz-muted text-xs italic p-4">No iteration convergence history available.</div>;
  }

  const margin = { top: 20, right: 20, bottom: 35, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const maxIter = history[history.length - 1].iteration;
  const residuals = history.map(h => Math.max(h.residualNorm, 1e-16));
  const logResiduals = residuals.map(r => Math.log10(r));
  const logMin = Math.min(...logResiduals);
  const logMax = Math.max(...logResiduals);
  const logRange = Math.max(logMax - logMin, 1);

  const points = history.map((step) => {
    const x = margin.left + (maxIter > 0 ? (step.iteration / maxIter) * w : 0);
    const y = margin.top + h - ((Math.log10(Math.max(step.residualNorm, 1e-16)) - logMin) / logRange) * h;
    return `${x},${y}`;
  });

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p}`).join(' ');

  const yTicks = [];
  for (let i = Math.floor(logMin); i <= Math.ceil(logMax); i++) {
    const y = margin.top + h - ((i - logMin) / logRange) * h;
    if (y >= margin.top && y <= margin.top + h) {
      yTicks.push({ y, label: `10^${i}` });
    }
  }

  const xTickCount = Math.min(6, maxIter);
  const xTicks = [];
  for (let i = 0; i <= xTickCount; i++) {
    const iter = Math.round((i / xTickCount) * maxIter);
    const x = margin.left + (iter / maxIter) * w;
    xTicks.push({ x, label: iter.toString() });
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ maxWidth: width }}>
      {/* Grid */}
      {yTicks.map((t, i) => (
        <line key={i} x1={margin.left} x2={margin.left + w} y1={t.y} y2={t.y} stroke="#E6E9EE" strokeDasharray="3 3" />
      ))}

      {/* Axes */}
      <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + h} stroke="#CBCBCB" strokeWidth="1.5" />
      <line x1={margin.left} x2={margin.left + w} y1={margin.top + h} y2={margin.top + h} stroke="#CBCBCB" strokeWidth="1.5" />

      {/* Y-axis labels */}
      {yTicks.map((t, i) => (
        <text key={i} x={margin.left - 8} y={t.y + 3} textAnchor="end" fill="#7E8694" fontSize="10" fontFamily="JetBrains Mono, monospace">
          {t.label}
        </text>
      ))}

      {/* X-axis labels */}
      {xTicks.map((t, i) => (
        <text key={i} x={t.x} y={margin.top + h + 18} textAnchor="middle" fill="#7E8694" fontSize="10" fontFamily="JetBrains Mono, monospace">
          {t.label}
        </text>
      ))}

      {/* Axis titles */}
      <text x={width / 2} y={height - 2} textAnchor="middle" fill="#4A4A4A" fontSize="11" fontWeight="600">
        Iteration
      </text>
      <text x={14} y={height / 2} textAnchor="middle" fill="#4A4A4A" fontSize="11" fontWeight="600" transform={`rotate(-90, 14, ${height / 2})`}>
        log₁₀(‖r‖₂)
      </text>

      {/* Data line with gradient */}
      <defs>
        <linearGradient id="convergenceGrad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#FFA896" />
          <stop offset="100%" stopColor="#68BA7F" />
        </linearGradient>
      </defs>
      <path d={pathD} fill="none" stroke="url(#convergenceGrad)" strokeWidth="2.5" strokeLinecap="round" />

      {/* Data points */}
      {history.length <= 60 && history.map((step, i) => {
        const x = margin.left + (maxIter > 0 ? (step.iteration / maxIter) * w : 0);
        const y = margin.top + h - ((Math.log10(Math.max(step.residualNorm, 1e-16)) - logMin) / logRange) * h;
        return <circle key={i} cx={x} cy={y} r={3} fill="#68BA7F" stroke="#FFFFFF" strokeWidth="1.5" />;
      })}
    </svg>
  );
};

/* ==================== VERIFICATION VIEW ==================== */
interface VerificationViewProps {
  Ax: Vector;
  b: Vector;
}

export const VerificationView: React.FC<VerificationViewProps> = ({ Ax, b }) => {
  const [page, setPage] = useState(0);
  const pageSize = 25;
  const n = Math.min(Ax.length, b.length);
  const totalPages = Math.ceil(n / pageSize);
  const start = page * pageSize;
  const end = Math.min(start + pageSize, n);

  return (
    <div className="w-full flex flex-col gap-3">
      <div className="verification-table-wrap w-full overflow-x-auto rounded-xl border">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="verification-table-head border-b">
              <th className="px-3 py-2 text-left font-bold">i</th>
              <th className="px-3 py-2 text-right text-[#276239] font-bold">(Ax)ᵢ</th>
              <th className="px-3 py-2 text-right font-bold">bᵢ</th>
              <th className="px-3 py-2 text-right text-[#DE6D55] font-bold">(Ax - b)ᵢ</th>
              <th className="px-3 py-2 text-center font-bold">Match</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: end - start }, (_, idx) => {
              const i = start + idx;
              const diff = Ax[i] - b[i];
              const absDiff = Math.abs(diff);
              const isMatch = absDiff < 1e-4;
              return (
                <tr key={i} className="verification-table-row border-b transition-colors">
                  <td className="px-3 py-1.5 viz-muted">i = {i + 1}</td>
                  <td className="px-3 py-1.5 text-right text-[#276239] font-medium">{Ax[i].toFixed(6)}</td>
                  <td className="px-3 py-1.5 text-right font-medium">{b[i].toFixed(6)}</td>
                  <td
                    className="px-3 py-1.5 text-right font-semibold"
                    style={{ color: absDiff < 1e-6 ? '#276239' : '#DE6D55' }}
                  >
                    {diff.toExponential(3)}
                  </td>
                  <td className="px-3 py-1.5 text-center">
                    <span
                      className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        isMatch
                          ? 'bg-[#CFFFDC] text-[#276239] border border-[#A3EFC0]'
                          : 'bg-[#FFA896]/30 text-[#DE6D55] border border-[#FFA896]'
                      }`}
                    >
                      {isMatch ? 'Exact ✓' : 'Diff ⚠'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs viz-muted px-1">
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            className="vector-pager px-2.5 py-1 border rounded-lg hover:bg-[var(--bg-hover)] disabled:opacity-40 transition-colors font-medium"
          >
            ← Prev
          </button>
          <span>Rows {start + 1}–{end} of {n}</span>
          <button
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="vector-pager px-2.5 py-1 border rounded-lg hover:bg-[var(--bg-hover)] disabled:opacity-40 transition-colors font-medium"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
};
