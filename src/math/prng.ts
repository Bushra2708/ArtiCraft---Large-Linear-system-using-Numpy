/**
 * Seedable PRNG (Mulberry32) for reproducible matrix generation.
 */
export class PRNG {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed >>> 0;
  }

  /**
   * Generates a 32-bit unsigned integer
   */
  nextUint32(): number {
    this.state |= 0;
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  }

  /**
   * Generates a float in [0, 1)
   */
  nextFloat(): number {
    return this.nextUint32() / 4294967296.0;
  }

  /**
   * Uniform float in range [min, max]
   */
  uniform(min: number = -10, max: number = 10): number {
    return min + this.nextFloat() * (max - min);
  }

  /**
   * Standard normal distribution via Box-Muller transform
   */
  normal(mean: number = 0, std: number = 1): number {
    let u = 0, v = 0;
    while (u === 0) u = this.nextFloat();
    while (v === 0) v = this.nextFloat();
    const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return mean + num * std;
  }
}
