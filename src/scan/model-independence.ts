/**
 * How much independent signal a model probability actually carries.
 *
 * The CLI trades `edge = model_prob - market_prob`, which assumes the two are
 * independent estimates. They are not. Octagon's model probability is
 * market-anchored: it starts from the contract's own debiased price and is then
 * shifted by research evidence, with the shift hard-capped by evidence grade
 * (A +/-20pp, B +/-12pp, C +/-7pp, D +/-3pp).
 *
 * Two consequences the backtest already showed without knowing the cause:
 * alpha versus an always-NO baseline sat near zero, and the largest edges
 * performed worst -- both are what you would expect if edge is a deterministic
 * function of price rather than a disagreement with it.
 *
 * The API now reports provenance, so an edge can be labelled instead of taken
 * at face value. Nothing here blocks a trade: the mandate does that. This
 * marks which signals are worth acting on.
 */

export type IndependenceLevel = 'independent' | 'bounded' | 'mechanical' | 'unknown';

export interface ModelProvenance {
  model_probability_source?: string | null;
  evidence_grade?: string | null;
}

/** Deviation each grade permits from the market-derived anchor, in points. */
export const GRADE_MAX_DEVIATION_PP: Record<string, number> = {
  A: 20,
  B: 12,
  C: 7,
  D: 3,
};

/**
 * Sources that carry no information beyond the market price. `market_baseline`
 * is the price re-expressed; `unmodeled` and `out_of_scope` were never
 * modelled; `determined` means the market has already settled.
 */
const MECHANICAL_SOURCES = new Set(['market_baseline', 'unmodeled', 'out_of_scope', 'determined']);

export function classifyIndependence(p: ModelProvenance | null | undefined): IndependenceLevel {
  if (!p) return 'unknown';
  const source = (p.model_probability_source ?? '').trim().toLowerCase();
  const grade = (p.evidence_grade ?? '').trim().toUpperCase();

  if (source && MECHANICAL_SOURCES.has(source)) return 'mechanical';
  if (!source && !grade) return 'unknown';
  // A low grade bounds the model to within a few points of the anchor, so the
  // "edge" it can express is mostly the debias curve, not research.
  if (grade === 'C' || grade === 'D') return 'bounded';
  if (grade === 'A' || grade === 'B') return 'independent';
  return 'unknown';
}

/** The largest edge the grade could physically produce, or null when unknown. */
export function maxExpressibleEdgePp(p: ModelProvenance | null | undefined): number | null {
  const grade = (p?.evidence_grade ?? '').trim().toUpperCase();
  return GRADE_MAX_DEVIATION_PP[grade] ?? null;
}

/**
 * True when the reported edge exceeds what the evidence grade allows the model
 * to express. That combination is not possible from research alone, so the
 * gap is coming from the debias curve — i.e. from the price itself.
 */
export function edgeExceedsEvidenceCap(edgePp: number, p: ModelProvenance | null | undefined): boolean {
  const cap = maxExpressibleEdgePp(p);
  return cap !== null && Math.abs(edgePp) > cap;
}

/** One line for `/analyze`, or null when there is nothing worth saying. */
export function formatIndependenceNote(
  edgePp: number | null,
  p: ModelProvenance | null | undefined,
): string | null {
  const level = classifyIndependence(p);
  const grade = (p?.evidence_grade ?? '').trim().toUpperCase();
  const source = (p?.model_probability_source ?? '').trim().toLowerCase();

  if (level === 'mechanical') {
    return `  ⚠ Edge is mechanical: model probability source is "${source}", which carries no information beyond the market price. Do not treat this edge as a disagreement.`;
  }
  if (level === 'bounded') {
    const cap = maxExpressibleEdgePp(p);
    const over = edgePp !== null && edgeExceedsEvidenceCap(edgePp, p);
    return (
      `  ⚠ Edge is weakly independent: evidence grade ${grade} caps the model at ±${cap}pp from the market-derived anchor` +
      (over ? `, yet the reported edge is ${Math.abs(edgePp!).toFixed(1)}pp — the excess comes from the debias curve, not research.` : '.')
    );
  }
  if (level === 'unknown') {
    return '  ⚠ Model provenance unavailable — cannot tell an independent estimate from the market price re-expressed.';
  }
  return null;
}
