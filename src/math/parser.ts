import type { Matrix, Vector } from './types';

export interface ParseResult {
  success: boolean;
  A?: Matrix;
  b?: Vector;
  error?: string;
  details?: string;
}

/**
 * Parses manual text or Python NumPy syntax into Matrix A and Vector b
 */
export function parseMatrixText(text: string): Matrix | { error: string } {
  const clean = text.trim();
  if (!clean) return { error: 'Input is empty.' };

  // Check if it is Python np.array syntax: np.array([[...], [...]])
  if (clean.includes('np.array') || clean.includes('array(')) {
    try {
      const match = clean.match(/\[.*\]/s);
      if (!match) return { error: 'Invalid Python array syntax: brackets not found.' };
      // Sanitize Python array to valid JSON
      const jsonStr = match[0]
        .replace(/\n/g, ' ')
        .replace(/,\s*]/g, ']')
        .replace(/,\s*,/g, ',')
        .replace(/\s+/g, ' ');
      const parsed = JSON.parse(jsonStr);
      if (Array.isArray(parsed) && Array.isArray(parsed[0])) {
        return parsed as Matrix;
      }
      return { error: 'Parsed Python array is not a 2D matrix.' };
    } catch {
      // Fallback to manual line parsing
    }
  }

  // Parse lines
  const lines = clean
    .split(/\r?\n|;/)
    .map(l => l.trim().replace(/^\[|\]$/g, '').trim())
    .filter(l => l.length > 0);

  if (lines.length === 0) return { error: 'No matrix rows detected.' };

  const matrix: Matrix = [];
  let colCount = -1;

  for (let r = 0; r < lines.length; r++) {
    const rawLine = lines[r];
    // Split on whitespace or commas
    const tokens = rawLine
      .split(/[\s,]+/)
      .map(t => t.trim())
      .filter(t => t.length > 0);

    if (tokens.length === 0) continue;

    const row: number[] = [];
    for (let c = 0; c < tokens.length; c++) {
      const val = parseFloat(tokens[c]);
      if (isNaN(val)) {
        return { error: `Invalid number "${tokens[c]}" at row ${r + 1}, column ${c + 1}.` };
      }
      row.push(val);
    }

    if (colCount === -1) {
      colCount = row.length;
    } else if (row.length !== colCount) {
      return {
        error: `Inconsistent row length: Row 1 has ${colCount} elements, but row ${r + 1} has ${row.length} elements.`
      };
    }

    matrix.push(row);
  }

  return matrix;
}

/**
 * Parses vector text (either 1 per line, comma-separated, space-separated, or np.array([...]))
 */
export function parseVectorText(text: string): Vector | { error: string } {
  const clean = text.trim();
  if (!clean) return { error: 'Vector input is empty.' };

  if (clean.includes('np.array') || clean.includes('array(')) {
    try {
      const match = clean.match(/\[.*\]/s);
      if (!match) return { error: 'Invalid Python array syntax: brackets not found.' };
      const jsonStr = match[0].replace(/\n/g, ' ').replace(/\s+/g, ' ');
      const parsed = JSON.parse(jsonStr);
      if (Array.isArray(parsed)) {
        // Flatten in case of [[1], [2]] column vector
        return parsed.flat().map(Number);
      }
    } catch {
      // Fallback
    }
  }

  const tokens = clean
    .replace(/^\[|\]$/g, '')
    .split(/[\s,;\n\r]+/)
    .map(t => t.trim())
    .filter(t => t.length > 0);

  if (tokens.length === 0) return { error: 'No vector elements found.' };

  const vec: number[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const val = parseFloat(tokens[i]);
    if (isNaN(val)) {
      return { error: `Invalid vector value "${tokens[i]}" at index ${i + 1}.` };
    }
    vec.push(val);
  }

  return vec;
}

/**
 * Validates dimensions of Matrix A and Vector b
 */
export function validateSystem(A: Matrix, b: Vector): { valid: boolean; error?: string; details?: string } {
  const m = A.length;
  if (m === 0) return { valid: false, error: 'Matrix A has no rows.' };
  const n = A[0]?.length || 0;
  if (n === 0) return { valid: false, error: 'Matrix A has no columns.' };

  if (b.length !== m) {
    return {
      valid: false,
      error: 'DIMENSION MISMATCH',
      details: `Matrix A has ${m} rows (${m}×${n}), but Vector b contains ${b.length} elements. Expected b ∈ ℝ^${m}.`
    };
  }

  return { valid: true };
}

/**
 * Parses algebraic linear equations like:
 *   2x + 3y - z = 5
 *   4x - y + 2z = 9
 *   -x + 2y + 2z = 4
 */
export function parseLinearEquations(text: string): { A: Matrix; b: Vector; varNames: string[] } | { error: string } {
  const lines = text
    .split(/\r?\n|;/)
    .map(l => l.trim())
    .filter(l => l.length > 0);

  if (lines.length === 0) {
    return { error: 'No equations entered.' };
  }

  // First pass: discover all variables in order of appearance
  const varSet = new Set<string>();
  const varList: string[] = [];

  for (const line of lines) {
    if (!line.includes('=')) {
      return { error: `Missing "=" in equation: "${line}"` };
    }
    const [lhs] = line.split('=');
    // Match identifiers like x, y, z, x1, x_2, alpha, etc.
    const varMatches = lhs.match(/[a-zA-Z][a-zA-Z0-9_]*/g) || [];
    for (const v of varMatches) {
      if (!varSet.has(v)) {
        varSet.add(v);
        varList.push(v);
      }
    }
  }

  if (varList.length === 0) {
    return { error: 'No variables (like x, y, z, or x1, x2) found in equations.' };
  }

  // Sort natural variables if they are numbered like x1, x2, x3
  const allNumbered = varList.every(v => /^x\d+$/i.test(v));
  if (allNumbered) {
    varList.sort((a, b) => parseInt(a.slice(1), 10) - parseInt(b.slice(1), 10));
  }

  const A: Matrix = [];
  const b: Vector = [];

  for (let r = 0; r < lines.length; r++) {
    const line = lines[r];
    const parts = line.split('=');
    if (parts.length !== 2) {
      return { error: `Invalid equation syntax on line ${r + 1}: contains multiple "=" signs.` };
    }
    const lhs = parts[0].trim();
    const rhsStr = parts[1].trim();

    const rhsVal = parseFloat(rhsStr);
    if (isNaN(rhsVal)) {
      return { error: `Right-hand side of equation "${line}" must be a number, got "${rhsStr}".` };
    }
    b.push(rhsVal);

    // Parse coefficients in LHS
    const row = new Array(varList.length).fill(0);

    // Normalize LHS: replace '-' with '+ -' to easily split by '+'
    // Keep unary leading minus intact
    let normalized = lhs.replace(/\s+/g, '');
    normalized = normalized.replace(/\*/g, ''); // remove multiplication *
    // insert + before any - that is not at the start
    normalized = normalized.replace(/([^-+])-/g, '$1+-');

    const terms = normalized.split('+').filter(t => t.length > 0);

    for (const term of terms) {
      // Find which variable this term belongs to
      let matchedVar: string | null = null;
      let coeffStr = '';

      for (const v of varList) {
        if (term.endsWith(v)) {
          // Check if it's the exact variable ending
          const prefix = term.slice(0, term.length - v.length);
          if (prefix === '' || prefix === '+' || prefix === '-' || !isNaN(parseFloat(prefix))) {
            matchedVar = v;
            coeffStr = prefix;
            break;
          }
        }
      }

      if (matchedVar) {
        let coeff = 1;
        if (coeffStr === '' || coeffStr === '+') coeff = 1;
        else if (coeffStr === '-') coeff = -1;
        else {
          const parsed = parseFloat(coeffStr);
          if (isNaN(parsed)) {
            return { error: `Could not parse coefficient "${coeffStr}" for variable "${matchedVar}" in "${term}".` };
          }
          coeff = parsed;
        }
        const colIdx = varList.indexOf(matchedVar);
        row[colIdx] += coeff;
      } else {
        // Constant on LHS: move to RHS by subtracting
        const constantVal = parseFloat(term);
        if (!isNaN(constantVal)) {
          b[b.length - 1] -= constantVal;
        } else {
          return { error: `Unrecognized term "${term}" in equation "${line}".` };
        }
      }
    }

    A.push(row);
  }

  return { A, b, varNames: varList };
}

