import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import type { Matrix } from '../../math/types';
import { ZoomIn, ZoomOut, RotateCcw, Crosshair } from 'lucide-react';

interface MatrixHeatmapCanvasProps {
  matrix: Matrix;
  onSelectRowCol?: (type: 'row' | 'col', index: number) => void;
  hoveredRow?: number | null;
  hoveredCol?: number | null;
  theme?: 'light' | 'dark';
}

export const MatrixHeatmapCanvas: React.FC<MatrixHeatmapCanvasProps> = ({
  matrix,
  onSelectRowCol,
  hoveredRow: externalHoveredRow,
  hoveredCol: externalHoveredCol,
  theme = 'light'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; val: number } | null>(null);

  const rows = matrix.length;
  const cols = matrix[0]?.length || 0;

  const isDark = theme === 'dark';

  const { minVal, maxVal, absMax } = useMemo(() => {
    let min = 0;
    let max = 0;
    for (let i = 0; i < rows; i++) {
      const row = matrix[i];
      for (let j = 0; j < cols; j++) {
        const v = row[j];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const absM = Math.max(Math.abs(min), Math.abs(max), 1e-6);
    return { minVal: min, maxVal: max, absMax: absM };
  }, [matrix, rows, cols]);

  // Color mapping with theme awareness
  const getColor = useCallback((val: number) => {
    // Zero baseline color:
    const zeroColor = isDark ? [32, 36, 42, 255] : [243, 245, 248, 255];
    if (Math.abs(val) < 1e-12) return zeroColor;

    const ratio = Math.min(1, Math.abs(val) / absMax);
    if (val > 0) {
      // Interpolate towards Pastel Sage [104, 186, 127]
      const r = Math.round(zeroColor[0] + (104 - zeroColor[0]) * ratio);
      const g = Math.round(zeroColor[1] + (186 - zeroColor[1]) * ratio);
      const b = Math.round(zeroColor[2] + (127 - zeroColor[2]) * ratio);
      return [r, g, b, 255];
    } else {
      // Interpolate towards Pastel Peach [255, 168, 150]
      const r = Math.round(zeroColor[0] + (255 - zeroColor[0]) * ratio);
      const g = Math.round(zeroColor[1] + (168 - zeroColor[1]) * ratio);
      const b = Math.round(zeroColor[2] + (150 - zeroColor[2]) * ratio);
      return [r, g, b, 255];
    }
  }, [absMax, isDark]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.save();

    // Canvas background
    ctx.fillStyle = isDark ? '#191C20' : '#FAFBFD';
    ctx.fillRect(0, 0, width, height);

    ctx.translate(offset.x, offset.y);
    ctx.scale(zoom, zoom);

    const margin = 32;
    const drawWidth = width - margin * 2;
    const drawHeight = height - margin * 2;
    const cellW = Math.max(1, drawWidth / cols);
    const cellH = Math.max(1, drawHeight / rows);

    if (rows * cols <= 10000) {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = matrix[i][j];
          const [r, g, b] = getColor(v);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(margin + j * cellW, margin + i * cellH, cellW - (cellW > 6 ? 1 : 0), cellH - (cellH > 6 ? 1 : 0));
        }
      }
    } else {
      const offscreen = document.createElement('canvas');
      offscreen.width = cols;
      offscreen.height = rows;
      const offCtx = offscreen.getContext('2d');
      if (offCtx) {
        const imgData = offCtx.createImageData(cols, rows);
        let ptr = 0;
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const [r, g, b, a] = getColor(matrix[i][j]);
            imgData.data[ptr++] = r;
            imgData.data[ptr++] = g;
            imgData.data[ptr++] = b;
            imgData.data[ptr++] = a;
          }
        }
        offCtx.putImageData(imgData, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreen, margin, margin, cols * cellW, rows * cellH);
      }
    }

    // Highlight row/column from hover or props
    const activeRow = hoveredCell ? hoveredCell.row : externalHoveredRow;
    const activeCol = hoveredCell ? hoveredCell.col : externalHoveredCol;

    if (activeRow !== undefined && activeRow !== null && activeRow >= 0 && activeRow < rows) {
      ctx.strokeStyle = '#68BA7F';
      ctx.lineWidth = 2 / zoom;
      ctx.strokeRect(margin, margin + activeRow * cellH, cols * cellW, cellH);
    }
    if (activeCol !== undefined && activeCol !== null && activeCol >= 0 && activeCol < cols) {
      ctx.strokeStyle = isDark ? '#D0D5DD' : '#4A4A4A';
      ctx.lineWidth = 2 / zoom;
      ctx.strokeRect(margin + activeCol * cellW, margin, cellW, rows * cellH);
    }

    // Border around matrix
    ctx.strokeStyle = isDark ? '#363C44' : '#CBCBCB';
    ctx.lineWidth = 1 / zoom;
    ctx.strokeRect(margin, margin, cols * cellW, rows * cellH);

    ctx.restore();
  }, [matrix, rows, cols, zoom, offset, getColor, hoveredCell, externalHoveredRow, externalHoveredCol, isDark]);

  useEffect(() => {
    let animationFrameId: number;
    const handleResize = () => {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = requestAnimationFrame(() => {
        if (containerRef.current && canvasRef.current) {
          const rect = containerRef.current.getBoundingClientRect();
          canvasRef.current.width = rect.width;
          canvasRef.current.height = rect.height;
          draw();
        }
      });
    };
    handleResize();
    const observer = new ResizeObserver(handleResize);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => {
      cancelAnimationFrame(animationFrameId);
      observer.disconnect();
    };
  }, [draw]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom(z => Math.max(0.2, Math.min(30, z * factor)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
      return;
    }

    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left - offset.x) / zoom;
    const mouseY = (e.clientY - rect.top - offset.y) / zoom;

    const margin = 32;
    const drawWidth = canvasRef.current.width - margin * 2;
    const drawHeight = canvasRef.current.height - margin * 2;
    const cellW = drawWidth / cols;
    const cellH = drawHeight / rows;

    const col = Math.floor((mouseX - margin) / cellW);
    const row = Math.floor((mouseY - margin) / cellH);

    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      setHoveredCell({ row, col, val: matrix[row][col] });
    } else {
      setHoveredCell(null);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleCanvasClick = () => {
    if (hoveredCell && onSelectRowCol) {
      onSelectRowCol('row', hoveredCell.row);
    }
  };

  const resetView = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[380px] select-none overflow-hidden rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-card-subtle)] shadow-sm"
    >
      <canvas
        ref={canvasRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { setIsDragging(false); setHoveredCell(null); }}
        onClick={handleCanvasClick}
        className="w-full h-full cursor-crosshair"
      />

      {/* Floating View Controls */}
      <div className="absolute top-3 right-3 flex items-center gap-1 p-1 bg-[var(--bg-card)]/90 backdrop-blur-md rounded-xl border border-[var(--border-subtle)] shadow-sm">
        <button
          onClick={() => setZoom(z => Math.min(30, z * 1.25))}
          className="p-1.5 text-[var(--text-body)] hover:bg-[var(--bg-hover)] rounded-lg transition-colors"
          title="Zoom In"
        >
          <ZoomIn size={15} />
        </button>
        <button
          onClick={() => setZoom(z => Math.max(0.2, z * 0.8))}
          className="p-1.5 text-[var(--text-body)] hover:bg-[var(--bg-hover)] rounded-lg transition-colors"
          title="Zoom Out"
        >
          <ZoomOut size={15} />
        </button>
        <button
          onClick={resetView}
          className="p-1.5 text-[var(--text-body)] hover:bg-[var(--bg-hover)] rounded-lg transition-colors"
          title="Reset View"
        >
          <RotateCcw size={15} />
        </button>
      </div>

      {/* Cell Hover Info Badge */}
      {hoveredCell && (
        <div className="absolute bottom-3 left-3 px-3.5 py-2 bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-mid)] rounded-xl shadow-md flex items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-1.5 text-[#68BA7F] font-semibold">
            <Crosshair size={13} />
            <span>A[{hoveredCell.row}, {hoveredCell.col}]</span>
          </div>
          <span className="text-[var(--text-main)] font-bold">
            {hoveredCell.val.toFixed(5)}
          </span>
          <button
            onClick={() => onSelectRowCol?.('row', hoveredCell.row)}
            className="text-[11px] text-[#68BA7F] hover:underline font-medium"
          >
            Inspect Row
          </button>
          <button
            onClick={() => onSelectRowCol?.('col', hoveredCell.col)}
            className="text-[11px] text-[var(--text-muted)] hover:underline font-medium"
          >
            Inspect Col
          </button>
        </div>
      )}

      {/* Heatmap Legend */}
      <div className="absolute bottom-3 right-3 flex items-center gap-2 px-3 py-1.5 bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-subtle)] rounded-xl text-[11px] font-mono text-[var(--text-body)] shadow-sm">
        <span className="text-[#FFA896] font-semibold">{minVal.toFixed(1)}</span>
        <div
          className="w-24 h-2.5 rounded-full border border-[var(--border-subtle)]"
          style={{
            background: isDark
              ? 'linear-gradient(to right, #FFA896 0%, #202428 50%, #68BA7F 100%)'
              : 'linear-gradient(to right, #FFA896 0%, #F1F3F6 50%, #68BA7F 100%)'
          }}
        />
        <span className="text-[#68BA7F] font-semibold">+{maxVal.toFixed(1)}</span>
      </div>
    </div>
  );
};
