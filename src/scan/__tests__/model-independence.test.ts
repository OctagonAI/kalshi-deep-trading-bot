import { describe, expect, test } from 'bun:test';
import {
  __setProvenanceSeen,
  apiServesProvenance,
  classifyIndependence,
  edgeExceedsEvidenceCap,
  formatIndependenceNote,
  maxExpressibleEdgePp,
  noteProvenanceObserved,
} from '../model-independence';

describe('classifyIndependence', () => {
  test('sources that carry no signal beyond price are mechanical', () => {
    for (const source of ['market_baseline', 'unmodeled', 'out_of_scope', 'determined']) {
      expect(classifyIndependence({ model_probability_source: source })).toBe('mechanical');
    }
  });

  test('low grades are bounded, high grades independent', () => {
    expect(classifyIndependence({ model_probability_source: 'recalibrated', evidence_grade: 'D' })).toBe('bounded');
    expect(classifyIndependence({ model_probability_source: 'recalibrated', evidence_grade: 'C' })).toBe('bounded');
    expect(classifyIndependence({ model_probability_source: 'recalibrated', evidence_grade: 'B' })).toBe('independent');
    expect(classifyIndependence({ model_probability_source: 'recalibrated', evidence_grade: 'A' })).toBe('independent');
  });

  test('absent provenance is unknown, not assumed good', () => {
    expect(classifyIndependence(null)).toBe('unknown');
    expect(classifyIndependence({})).toBe('unknown');
  });
});

describe('evidence caps', () => {
  test('caps match the documented deviation limits', () => {
    expect(maxExpressibleEdgePp({ evidence_grade: 'A' })).toBe(20);
    expect(maxExpressibleEdgePp({ evidence_grade: 'D' })).toBe(3);
    expect(maxExpressibleEdgePp({})).toBeNull();
  });

  test('an edge larger than the grade allows cannot have come from research', () => {
    // Grade D bounds the model to ±3pp of the anchor, so a 30pp edge is the
    // debias curve, not evidence.
    expect(edgeExceedsEvidenceCap(30, { evidence_grade: 'D' })).toBe(true);
    expect(edgeExceedsEvidenceCap(-30, { evidence_grade: 'D' })).toBe(true);
    expect(edgeExceedsEvidenceCap(2, { evidence_grade: 'D' })).toBe(false);
    // An unknown grade cannot be judged, so it is not flagged.
    expect(edgeExceedsEvidenceCap(30, {})).toBe(false);
  });
});

describe('formatIndependenceNote', () => {
  test('mechanical edges are called out plainly', () => {
    const note = formatIndependenceNote(12, { model_probability_source: 'market_baseline' });
    expect(note).toContain('mechanical');
    expect(note).toContain('market_baseline');
  });

  test('a bounded edge over its cap says where the excess came from', () => {
    const note = formatIndependenceNote(30, { model_probability_source: 'recalibrated', evidence_grade: 'D' })!;
    expect(note).toContain('±3pp');
    expect(note).toContain('debias curve');
  });

  test('a well-evidenced edge gets no warning', () => {
    expect(formatIndependenceNote(8, { model_probability_source: 'recalibrated', evidence_grade: 'A' })).toBeNull();
  });

  test('stays silent about missing provenance until the API is seen serving it', () => {
    // The fields ship in a later API release. Warning on every analysis until
    // then would train the reader to ignore the warning, and it would still be
    // firing on the day it starts meaning something.
    __setProvenanceSeen(false);
    expect(formatIndependenceNote(8, null)).toBeNull();
    expect(formatIndependenceNote(8, {})).toBeNull();
  });

  test('once provenance is observed, its absence becomes reportable', () => {
    __setProvenanceSeen(false);
    noteProvenanceObserved({ model_probability_source: 'recalibrated', evidence_grade: 'B' });
    expect(apiServesProvenance()).toBe(true);
    expect(formatIndependenceNote(8, null)).toContain('provenance unavailable');
    __setProvenanceSeen(false);
  });

  test('all-null provenance does not count as the API serving it', () => {
    __setProvenanceSeen(false);
    noteProvenanceObserved({ model_probability_source: null, evidence_grade: null });
    expect(apiServesProvenance()).toBe(false);
  });
});

describe('a known source without a grade', () => {
  test('is not reported as missing provenance', () => {
    // Live prod shape: settled rungs come back `recalibrated` with a null grade
    // because they were never researched. Calling that "provenance unavailable"
    // is false -- the source is present and says the value came from the model.
    __setProvenanceSeen(true);
    const p = { model_probability_source: 'recalibrated', evidence_grade: null };
    expect(classifyIndependence(p)).toBe('ungraded');
    const note = formatIndependenceNote(9, p);
    expect(note).not.toBeNull();
    expect(note).not.toContain('provenance unavailable');
    expect(note).toContain('unbounded by evidence');
  });

  test('still reports genuinely absent provenance as unavailable', () => {
    __setProvenanceSeen(true);
    const note = formatIndependenceNote(9, { model_probability_source: null, evidence_grade: null });
    expect(note).toContain('provenance unavailable');
  });

  test('does not override a mechanical source that has no grade', () => {
    // market_baseline never carries a grade; it must stay mechanical, which is
    // the stronger and more actionable statement.
    expect(classifyIndependence({ model_probability_source: 'market_baseline', evidence_grade: null })).toBe('mechanical');
  });
});
