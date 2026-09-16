import React, { useRef, useEffect, useCallback, useMemo } from 'react';
import type { Matrix } from '../../math/types';

interface SparsityCanvasProps {
  matrix: Matrix;
  theme?: 'light' | 'dark';
}

export const SparsityCanvas: React.FC<SparsityCanvasProps> = ({ matrix, theme = 'light' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const rows = matrix.length;
  const cols = matrix[0]?.length || 0;
  const isDark = theme === 'dark';

  const stats = useMemo(() => {
    let nonZero = 0;
    const total = rows * cols;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (Math.abs(matrix[i][j]) > 1e-12) nonZero++;
      }
    }
    return {
      nonZero,
      total,
      sparsity: total > 0 ? ((total - nonZero) / total * 100) : 0
    };
  }, [matrix, rows, cols]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = isDark ? '#191C20' : '#FAFBFD';
    ctx.fillRect(0, 0, w, h);

    const margin = 28;
    const dw = w - margin * 2;
    const dh = h - margin * 2;
    const cellW = dw / cols;
    const cellH = dh / rows;

    if (rows * cols <= 50000) {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const isNonZero = Math.abs(matrix[i][j]) > 1e-12;
          if (isNonZero) {
            ctx.fillStyle = '#68BA7F';
            ctx.fillRect(
              margin + j * cellW,
              margin + i * cellH,
              Math.max(1.5, cellW - (cellW > 4 ? 0.5 : 0)),
              Math.max(1.5, cellH - (cellH > 4 ? 0.5 : 0))
            );
          }
        }
      }
    } else {
      const offscreen = document.createElement('canvas');
      offscreen.width = cols;
      offscreen.height = rows;
      const offCtx = offscreen.getContext('2d');
      if (offCtx) {
        const img = offCtx.createImageData(cols, rows);
        let ptr = 0;
        const bgR = isDark ? 25 : 250;
        const bgG = isDark ? 28 : 251;
        const bgB = isDark ? 32 : 253;
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const nz = Math.abs(matrix[i][j]) > 1e-12;
            img.data[ptr++] = nz ? 104 : bgR;
            img.data[ptr++] = nz ? 186 : bgG;
            img.data[ptr++] = nz ? 127 : bgB;
            img.data[ptr++] = 255;
          }
        }
        offCtx.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreen, margin, margin, dw, dh);
      }
    }

    ctx.strokeStyle = isDark ? '#363C44' : '#CBCBCB';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin, margin, dw, dh);
  }, [matrix, rows, cols, isDark]);

  useEffect(() => {
    let animationFrameId: number;
    const handleResize = () => {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = requestAnimationFrame(() => {
        if (containerRef.current && canvasRef.current) {
          const r = containerRef.current.getBoundingClientRect();
          canvasRef.current.width = r.width;
          canvasRef.current.height = r.height;
          draw();
        }
      });
    };
    handleResize();
    const obs = new ResizeObserver(handleResize);
    if (containerRef.current) obs.observe(containerRef.current);
    return () => {
      cancelAnimationFrame(animationFrameId);
      obs.disconnect();
    };
  }, [draw]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <div className="w-full flex flex-col gap-3">
      <div
        ref={containerRef}
        className="relative w-full h-[320px] bg-[var(--bg-card-subtle)] rounded-2xl border border-[var(--border-subtle)] overflow-hidden shadow-sm"
      >
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
      <div className="flex items-center justify-between text-xs font-mono px-3 py-2 bg-[var(--bg-card)] rounded-xl border border-[var(--border-subtle)] shadow-sm">
        <div className="flex items-center gap-5">
          <span className="text-[var(--text-body)]">
            Non-zeros: <strong className="text-[#68BA7F] font-bold">{stats.nonZero.toLocaleString()}</strong>
          </span>
          <span className="text-[var(--text-muted)]">
            Total: <strong className="text-[var(--text-body)]">{stats.total.toLocaleString()}</strong>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[var(--text-muted)]">Sparsity:</span>
          <span className="px-2.5 py-0.5 rounded-full bg-[var(--c-sage-soft)] text-[#68BA7F] font-bold border border-[var(--c-sage-border)]">
            {stats.sparsity.toFixed(2)}%
          </span>
        </div>
      </div>
    </div>
  );
};
