export type OctagonVariant = 'default' | 'cache' | 'refresh';

export type MispricingSignal = 'overpriced' | 'underpriced' | 'fair_value';

export type OctagonInvoker = (ticker: string, variant: OctagonVariant) => Promise<string>;

export type DriverCategory = 'political' | 'economic' | 'sentiment' | 'technical';
export type DriverImpact = 'high' | 'medium' | 'low';

export interface PriceDriver {
  claim: string;
  category: DriverCategory;
  impact: DriverImpact;
  sourceUrl?: string;
}

export interface Catalyst {
  date: string; // ISO date
  event: string;
  impact: DriverImpact;
  potentialMove: string;
}

export interface Source {
  url: string;
  title?: string;
}

export interface OctagonReport {
  /**
   * Provenance of modelProb. Absent on older cached rows. Without it a
   * consumer cannot tell an independent estimate from the market price
   * re-expressed -- and this CLI trades the difference between the two.
   */
  provenance?: { model_probability_source?: string | null; evidence_grade?: string | null } | null;
  ticker: string;
  eventTicker: string;
  modelProb: number;
  marketProb: number;
  mispricingSignal: MispricingSignal;
  drivers: PriceDriver[];
  catalysts: Catalyst[];
  sources: Source[];
  resolutionHistory: string;
  contractSnapshot: string;
  variantUsed: OctagonVariant;
  fetchedAt: number; // epoch seconds
  rawResponse: string; // full raw Octagon API response
  cacheMiss: boolean; // true when cache returned no meaningful data
  reportId: string; // persisted DB report_id (UUID)
}

export type ConfidenceLevel = 'low' | 'moderate' | 'high' | 'very_high';

export interface EdgeSnapshot {
  ticker: string;            // market ticker e.g. 'KXBTC-26MAR-B80000'
  eventTicker: string;       // event ticker e.g. 'KXBTC-26MAR'
  modelProb: number;         // Octagon probability [0, 1]
  marketProb: number;        // Kalshi mid price as probability [0, 1]
  edge: number;              // modelProb - marketProb (signed)
  confidence: ConfidenceLevel;
  drivers: PriceDriver[];
  catalysts: Catalyst[];
  sources: Source[];
  octagonReportId: string;
  cacheHit: boolean;
  timestamp: number;         // epoch seconds
}
